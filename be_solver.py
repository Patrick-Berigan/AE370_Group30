"""
be_solver.py  --  the numerical method ONLY (no problem-specific numbers here).

1D transient heat conduction through stacked layers, finite volumes (cell-centered),
advanced in time with Backward Euler.

Boundary conditions:
    x = 0 (hot face): prescribed heat flux q(t) [W/m^2] into the first cell
    x = L (back face): insulated (no energy leaves the system)

Semi-discrete system (one ODE per cell):
    C_i dT_i/dt = G_{i-1/2} (T_{i-1} - T_i) + G_{i+1/2} (T_{i+1} - T_i) + [q(t) if i == 0]
which we write as   dT/dt = A T + b(t)   with   A = C^{-1} K.

Backward Euler step:
    (I - dt*A) T_{n+1} = T_n + dt*b(t_{n+1})
"""
import numpy as np
from scipy.linalg import solve_banded


# ---------------------------------------------------------------------------
# 1. Mesh
# ---------------------------------------------------------------------------
def build_mesh(layers, dx):
    """
    layers : list of (thickness [m], k [W/m/K], rho [kg/m^3], cp [J/kg/K]), hot side first
    dx     : fixed cell width [m]; every layer thickness must be a whole multiple of dx

    Returns dict with arrays (one entry per cell):
        x (cell centers), dx, k, rho, cp, layer_id, plus N (number of cells), L (total thickness)
    """
    k, rho, cp, layer_id = [], [], [], []
    for j, (thick, kj, rhoj, cpj) in enumerate(layers):
        n = int(round(thick / dx))
        if n < 1 or abs(n * dx - thick) > 1e-6 * thick:
            raise ValueError(f"Layer {j}: thickness {thick} m is not a whole multiple of dx = {dx} m")
        k += [kj] * n
        rho += [rhoj] * n
        cp += [cpj] * n
        layer_id += [j] * n
    N = len(k)
    return {
        "x": (np.arange(N) + 0.5) * dx,
        "dx": np.full(N, dx),
        "k": np.array(k, dtype=float),
        "rho": np.array(rho, dtype=float),
        "cp": np.array(cp, dtype=float),
        "layer_id": np.array(layer_id, dtype=int),
        "N": N,
        "L": N * dx,
    }


# ---------------------------------------------------------------------------
# 2. Conductances between neighboring cells (series resistance of two half cells)
# ---------------------------------------------------------------------------
def build_conductances(dx, k):
    """G[i] connects cell i and i+1 (length N-1), units W/m^2/K.
    Same formula inside a layer and across a material border."""
    return 1.0 / (dx[:-1] / (2.0 * k[:-1]) + dx[1:] / (2.0 * k[1:]))


# ---------------------------------------------------------------------------
# 3. Heat capacity per unit area and the matrix A (banded storage)
# ---------------------------------------------------------------------------
def build_C(dx, rho, cp):
    """Heat capacity per unit area of each cell, J/m^2/K."""
    return rho * cp * dx


def build_A(C, G):
    """
    A = C^{-1} K as three bands (insulated back face, so nothing extra at the last cell):
        upper[i] = A[i, i+1],  diag[i] = A[i, i],  lower[i] = A[i+1, i]   (upper/lower length N-1)
    """
    N = len(C)
    diag = np.zeros(N)
    diag[:-1] -= G / C[:-1]
    diag[1:] -= G / C[1:]
    upper = G / C[:-1]
    lower = G / C[1:]
    return upper, diag, lower


# ---------------------------------------------------------------------------
# 4. Forcing vector b(t): hot-face flux goes into cell 0 only
# ---------------------------------------------------------------------------
def build_b(t, C, q_func):
    b = np.zeros(len(C))
    b[0] = q_func(t) / C[0]
    return b


# ---------------------------------------------------------------------------
# 5. Left-hand-side matrix  (I - dt*A)  in the banded form solve_banded wants
# ---------------------------------------------------------------------------
def build_lhs_banded(A_bands, dt):
    upper, diag, lower = A_bands
    N = len(diag)
    ab = np.zeros((3, N))
    ab[0, 1:] = -dt * upper
    ab[1, :] = 1.0 - dt * diag
    ab[2, :-1] = -dt * lower
    return ab


# ---------------------------------------------------------------------------
# 6. Linear solve (tridiagonal)
# ---------------------------------------------------------------------------
def solve_tridiagonal(ab, rhs):
    return solve_banded((1, 1), ab, rhs)


# ---------------------------------------------------------------------------
# 7. One Backward Euler step  t_n -> t_{n+1}
# ---------------------------------------------------------------------------
def backward_euler_step(lhs_banded, T_n, b_next, dt):
    return solve_tridiagonal(lhs_banded, T_n + dt * b_next)


# ---------------------------------------------------------------------------
# 8. Time marching loop
# ---------------------------------------------------------------------------
def time_march(mesh, q_func, T0, dt, t_end, store_every=1):
    """
    March from t = 0 to t_end.  dt is nudged slightly so that t_end is hit exactly.
    store_every : keep every n-th step (the first and last step are always kept).

    Returns t (stored times), T_hist (stored times x N), and the dt actually used.
    """
    C = build_C(mesh["dx"], mesh["rho"], mesh["cp"])
    G = build_conductances(mesh["dx"], mesh["k"])
    A_bands = build_A(C, G)

    n_steps = max(1, int(np.ceil(t_end / dt - 1e-9)))
    dt = t_end / n_steps
    lhs = build_lhs_banded(A_bands, dt)       # constant matrix: build once

    T = np.full(mesh["N"], float(T0))
    t_list, T_list = [0.0], [T.copy()]
    for n in range(1, n_steps + 1):
        t_next = n * dt
        T = backward_euler_step(lhs, T, build_b(t_next, C, q_func), dt)
        if n % store_every == 0 or n == n_steps:
            t_list.append(t_next)
            T_list.append(T.copy())
    return np.array(t_list), np.array(T_list), dt
