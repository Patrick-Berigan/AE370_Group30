"""
verify.py  --  confirmation simulation: ONE material, constant properties, constant flux,
insulated back face.  Thick wall + short run => behaves like a semi-infinite solid, so the
exact solution applies.

Run:  python verify.py
"""
import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

from be_solver import build_mesh, build_C, time_march

# ---- verification problem (simple placeholder material) -------------------
K, RHO, CP = 1.0, 1000.0, 1000.0           # W/m/K, kg/m^3, J/kg/K
ALPHA = K / (RHO * CP)                      # 1e-6 m^2/s
Q = 1.0e4                                   # W/m^2
T0 = 300.0                                  # K
L = 0.10                                    # m   (thermal depth sqrt(alpha t) ~ 7 mm << L)
T_END = 50.0                                # s
LAYERS = [(L, K, RHO, CP)]


def q_const(t):
    return Q


# ---------------------------------------------------------------------------
# 8. Exact solution: semi-infinite solid, constant surface flux
# ---------------------------------------------------------------------------
def exact_semi_infinite(x, t, q, k, alpha, T0):
    return T0 + (2 * q / k) * np.sqrt(alpha * t / np.pi) * np.exp(-x**2 / (4 * alpha * t)) \
              - (q * x / k) * erfc(x / (2 * np.sqrt(alpha * t)))


# ---------------------------------------------------------------------------
# 9. Error norms
# ---------------------------------------------------------------------------
def error_norm(T_num, T_ref, kind="max"):
    e = T_num - T_ref
    if kind == "max":
        return np.max(np.abs(e))
    return np.sqrt(np.mean(e**2))           # discrete L2 (RMS)


# ---------------------------------------------------------------------------
# 10. Energy conservation check (insulated back face, constant flux)
# ---------------------------------------------------------------------------
def energy_check(mesh, T_final, T0, q, t_end):
    """Relative difference between stored energy and energy put in at x = 0."""
    C = build_C(mesh["dx"], mesh["rho"], mesh["cp"])
    stored = np.sum(C * (T_final - T0))
    supplied = q * t_end
    return (stored - supplied) / supplied


# ---------------------------------------------------------------------------
# 13. Slope of log(err) vs log(h)
# ---------------------------------------------------------------------------
def fit_slope(h_list, err_list):
    return np.polyfit(np.log(h_list), np.log(err_list), 1)[0]


# ---------------------------------------------------------------------------
# 11. Temporal convergence: same mesh, compare to a tiny-dt run on that mesh
# ---------------------------------------------------------------------------
def temporal_convergence(dx, dt_list, dt_ref):
    mesh = build_mesh(LAYERS, dx)
    _, Tref, _ = time_march(mesh, q_const, T0, dt_ref, T_END, store_every=10**9)
    errs = []
    for dt in dt_list:
        _, T, _ = time_march(mesh, q_const, T0, dt, T_END, store_every=10**9)
        errs.append(error_norm(T[-1], Tref[-1]))
    return np.array(errs)


# ---------------------------------------------------------------------------
# 12. Spatial convergence vs the exact solution; dt shrinks like dx^2 so the
#     first-order time error shrinks as fast as the second-order space error.
# ---------------------------------------------------------------------------
def spatial_convergence(N_list, dt_coeff=0.05):
    errs, dxs = [], []
    for N in N_list:
        dx = L / N
        dt = dt_coeff * dx**2 / ALPHA
        mesh = build_mesh(LAYERS, dx)
        _, T, _ = time_march(mesh, q_const, T0, dt, T_END, store_every=10**9)
        Tex = exact_semi_infinite(mesh["x"], T_END, Q, K, ALPHA, T0)
        errs.append(error_norm(T[-1], Tex))
        dxs.append(dx)
    return np.array(dxs), np.array(errs)


def main():
    # ---- (a) numerical vs exact profile and energy check ------------------
    dx = 0.5e-3
    mesh = build_mesh(LAYERS, dx)
    _, T, dt_used = time_march(mesh, q_const, T0, 0.01, T_END, store_every=10**9)
    Tex = exact_semi_infinite(mesh["x"], T_END, Q, K, ALPHA, T0)
    rise = Tex[0] - T0
    print(f"[profile] dx = {dx*1e3:.2f} mm, dt = {dt_used:.4f} s")
    print(f"[profile] max |T_num - T_exact| = {error_norm(T[-1], Tex):.4e} K "
          f"({100*error_norm(T[-1], Tex)/rise:.3f}% of the {rise:.1f} K surface rise)")
    print(f"[energy ] relative energy imbalance = {energy_check(mesh, T[-1], T0, Q, T_END):.3e}")
    print(f"[back   ] back-face rise = {T[-1][-1]-T0:.3e} K (should be ~0)")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(mesh["x"] * 1e3, Tex, "k-", label="exact (semi-infinite)")
    ax.plot(mesh["x"][::6] * 1e3, T[-1][::6], "o", mfc="none", label="backward Euler")
    ax.set_xlim(0, 40); ax.set_xlabel("x [mm]"); ax.set_ylabel("T [K]")
    ax.set_title(f"Verification profile at t = {T_END:.0f} s"); ax.legend()
    fig.tight_layout(); fig.savefig("figs/verify_profile.png", dpi=200)

    # ---- (b) temporal convergence -----------------------------------------
    dt_list = np.array([5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625])
    errs_t = temporal_convergence(1.0e-3, dt_list, dt_ref=0.0025)
    s_t = fit_slope(dt_list, errs_t)
    print("\n[time] dt           error [K]")
    for d, e in zip(dt_list, errs_t):
        print(f"[time] {d:9.5f}   {e:.4e}")
    print(f"[time] fitted slope = {s_t:.3f}   (expected 1)")

    # ---- (c) spatial convergence ------------------------------------------
    N_list = [25, 50, 100, 200, 400]
    dxs, errs_x = spatial_convergence(N_list)
    s_x = fit_slope(dxs, errs_x)
    print("\n[space] dx [m]       error [K]")
    for d, e in zip(dxs, errs_x):
        print(f"[space] {d:.3e}   {e:.4e}")
    print(f"[space] fitted slope = {s_x:.3f}   (expected 2)")

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    axs[0].loglog(dt_list, errs_t, "o-", label="error")
    axs[0].loglog(dt_list, errs_t[0] * dt_list / dt_list[0], "k--", label="slope 1")
    axs[0].set_xlabel(r"$\Delta t$ [s]"); axs[0].set_ylabel("max error [K]")
    axs[0].set_title(f"Time (slope {s_t:.2f})"); axs[0].legend()
    axs[1].loglog(dxs, errs_x, "o-", label="error vs exact")
    axs[1].loglog(dxs, errs_x[0] * (dxs / dxs[0])**2, "k--", label="slope 2")
    axs[1].set_xlabel(r"$\Delta x$ [m]"); axs[1].set_ylabel("max error [K]")
    axs[1].set_title(f"Space (slope {s_x:.2f})"); axs[1].legend()
    fig.tight_layout(); fig.savefig("figs/verify_convergence.png", dpi=200)


if __name__ == "__main__":
    main()
