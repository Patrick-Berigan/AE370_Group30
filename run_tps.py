"""
run_tps.py  --  three-layer thermal protection system (TPS) case, results and parameter studies.

Layers (hot side first). ALL NUMBERS ARE BALLPARK PLACEHOLDERS - replace with sourced values.
Each layer thickness must be a whole multiple of DX.

Run:  python run_tps.py
"""
import copy
import numpy as np
import matplotlib.pyplot as plt

from be_solver import build_mesh, time_march

# thickness [m], k [W/m/K], rho [kg/m^3], cp [J/kg/K]
LAYER_NAMES = ["Ablative liner", "EPDM insulation", "Steel casing"]
BASE = {
    "layers": [
        (3.0e-3, 0.50, 1450.0, 1400.0),   # carbon-phenolic-like liner
        (4.0e-3, 0.25, 1100.0, 2000.0),   # EPDM-rubber-like insulation
        (5.0e-3, 30.0, 7800.0,  500.0),   # steel casing
    ],
    "dx": 0.25e-3,        # m   fixed everywhere
    "q": 1.0e5,           # W/m^2 constant hot-face flux
    "T0": 293.0,          # K
    "t_end": 120.0,       # s   operating duration
    "dt": 0.05,           # s
}
CASING = 2                # index of the casing layer


# ---------------------------------------------------------------------------
# 14. Run one TPS case
# ---------------------------------------------------------------------------
def run_tps_case(params, store_every=1):
    mesh = build_mesh(params["layers"], params["dx"])
    q = params["q"]
    t, T_hist, dt_used = time_march(mesh, lambda tt: q, params["T0"], params["dt"],
                                    params["t_end"], store_every=store_every)
    return mesh, t, T_hist


# ---------------------------------------------------------------------------
# 15. Peak casing temperature (max over casing cells and over time)
# ---------------------------------------------------------------------------
def max_casing_temp(mesh, T_hist, casing=CASING):
    return T_hist[:, mesh["layer_id"] == casing].max()


# ---------------------------------------------------------------------------
# 16. Thermal lag: first time the casing's hot-side cell rises by dT_threshold
#     (linear interpolation between stored times). Returns nan if never reached.
# ---------------------------------------------------------------------------
def thermal_lag(mesh, t, T_hist, T0, dT_threshold, casing=CASING):
    idx = np.where(mesh["layer_id"] == casing)[0][0]
    rise = T_hist[:, idx] - T0
    above = np.where(rise >= dT_threshold)[0]
    if len(above) == 0:
        return np.nan
    j = above[0]
    if j == 0:
        return t[0]
    return t[j - 1] + (dT_threshold - rise[j - 1]) / (rise[j] - rise[j - 1]) * (t[j] - t[j - 1])


# ---------------------------------------------------------------------------
# helper: change one parameter by name
#   "q", "t_end", "thickness:<layer index>", "k:<layer index>"
# ---------------------------------------------------------------------------
def apply_param(params, name, value):
    p = copy.deepcopy(params)
    if name in ("q", "t_end", "dt", "dx", "T0"):
        p[name] = value
    elif name.startswith("thickness:") or name.startswith("k:"):
        kind, j = name.split(":")
        j = int(j)
        L = list(p["layers"][j])
        L[0 if kind == "thickness" else 1] = value
        p["layers"][j] = tuple(L)
    else:
        raise ValueError(name)
    return p


# ---------------------------------------------------------------------------
# 17. Sweep a parameter; report peak casing temperature, casing-front rise, thermal lag
# ---------------------------------------------------------------------------
def sweep(base, name, values, dT_threshold=10.0):
    out = []
    for v in values:
        p = apply_param(base, name, v)
        mesh, t, T = run_tps_case(p)
        out.append({
            "value": v,
            "T_casing_max": max_casing_temp(mesh, T),
            "T_hot_face": T[-1, 0],
            "lag": thermal_lag(mesh, t, T, p["T0"], dT_threshold),
        })
    return out


# ---------------------------------------------------------------------------
# 18. Largest dt (from a list) whose peak casing temperature is within tol [K]
#     of a very fine reference run.
# ---------------------------------------------------------------------------
def find_required_dt(params, dt_list, dt_ref, tol=0.1):
    pr = copy.deepcopy(params); pr["dt"] = dt_ref
    mesh, _, Tr = run_tps_case(pr, store_every=10**9)
    ref = max_casing_temp(mesh, Tr)
    rows = []
    for dt in sorted(dt_list, reverse=True):
        p = copy.deepcopy(params); p["dt"] = dt
        mesh, _, T = run_tps_case(p, store_every=10**9)
        rows.append((dt, abs(max_casing_temp(mesh, T) - ref)))
    ok = [dt for dt, e in rows if e < tol]
    return (max(ok) if ok else None), rows, ref


# ---------------------------------------------------------------------------
# 19. Plots
# ---------------------------------------------------------------------------
def plot_profiles(mesh, t, T, fname="figs/tps_profiles.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for frac in (0.1, 0.25, 0.5, 1.0):
        j = np.argmin(np.abs(t - frac * t[-1]))
        ax.plot(mesh["x"] * 1e3, T[j], label=f"t = {t[j]:.0f} s")
    edges = np.cumsum([l[0] for l in BASE["layers"]]) * 1e3
    for e in edges[:-1]:
        ax.axvline(e, color="gray", ls=":")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("T [K]"); ax.legend()
    ax.set_title("Temperature through the wall (dotted = layer borders)")
    fig.tight_layout(); fig.savefig(fname, dpi=200)


def plot_history(mesh, t, T, fname="figs/tps_history.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for j, name in enumerate(LAYER_NAMES):
        cells = np.where(mesh["layer_id"] == j)[0]
        ax.plot(t, T[:, cells[0]], label=f"{name}, hot-side cell")
    ax.plot(t, T[:, -1], "k--", label="Back face")
    ax.set_xlabel("t [s]"); ax.set_ylabel("T [K]"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fname, dpi=200)


def plot_sweep(rows, xlabel, fname, xscale=1.0):
    x = [r["value"] * xscale for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(x, [r["T_casing_max"] for r in rows], "o-")
    ax.set_xlabel(xlabel); ax.set_ylabel("Peak casing temperature [K]")
    fig.tight_layout(); fig.savefig(fname, dpi=200)


def plot_dt_study(rows, tol, fname="figs/tps_dt_study.png"):
    dts = [r[0] for r in rows]; errs = [max(r[1], 1e-16) for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.loglog(dts, errs, "o-")
    ax.axhline(tol, color="r", ls="--", label=f"tolerance {tol} K")
    ax.set_xlabel(r"$\Delta t$ [s]"); ax.set_ylabel("Error in peak casing T [K]"); ax.legend()
    fig.tight_layout(); fig.savefig(fname, dpi=200)


def main():
    # ---- baseline run -----------------------------------------------------
    mesh, t, T = run_tps_case(BASE)
    print(f"Cells: {mesh['N']}, total thickness {mesh['L']*1e3:.1f} mm")
    print(f"Hot-face T at end: {T[-1,0]:.1f} K,  peak casing T: {max_casing_temp(mesh, T):.3f} K")
    print(f"Thermal lag (casing +10 K): {thermal_lag(mesh, t, T, BASE['T0'], 10.0):.1f} s")
    plot_profiles(mesh, t, T); plot_history(mesh, t, T)

    # ---- time-step requirement -------------------------------------------
    tol = 0.1
    dt_list = [10, 5, 2.5, 1, 0.5, 0.25, 0.1, 0.05]
    dt_ok, rows, ref = find_required_dt(BASE, dt_list, dt_ref=0.005, tol=tol)
    print(f"\nReference peak casing T (dt = 0.005 s): {ref:.4f} K")
    for dt, e in rows:
        print(f"  dt = {dt:7.3f} s   error = {e:.3e} K")
    print(f"Largest dt within {tol} K: {dt_ok}")
    plot_dt_study(rows, tol)

    # ---- parameter sweeps -------------------------------------------------
    res = sweep(BASE, "thickness:1", np.array([2, 3, 4, 5, 6, 8]) * 1e-3)
    print("\nInsulation thickness sweep [mm -> peak casing T, lag (+10 K)]")
    for r in res:
        print(f"  {r['value']*1e3:5.1f}  {r['T_casing_max']:.2f} K   lag = {r['lag']}")
    plot_sweep(res, "Insulation thickness [mm]", "figs/sweep_thickness.png", 1e3)

    res = sweep(BASE, "q", [0.5e5, 0.75e5, 1e5, 1.25e5, 1.5e5])
    plot_sweep(res, r"Hot-face flux [kW/m$^2$]", "figs/sweep_flux.png", 1e-3)

    res = sweep(BASE, "t_end", [60, 90, 120, 150, 180, 240])
    plot_sweep(res, "Operating duration [s]", "figs/sweep_duration.png")

    res = sweep(BASE, "k:1", [0.15, 0.2, 0.25, 0.3, 0.4])
    plot_sweep(res, "Insulation k [W/m/K]", "figs/sweep_k.png")


if __name__ == "__main__":
    main()
