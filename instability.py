#%% Import libraries

import numpy as np
import h5py as h5
from scipy.integrate import solve_ivp
from modules.gensolver import save_data
from modules.triads import make_triad
import os

#%% Parameters

# Npx,Npy=512,512
Npx,Npy=1024,1024
Lx,Ly=32*np.pi,32*np.pi

# Allowed values: "phi_pump", "P_pump", "both_same_phase", "both_different_phase"
INIT_CASE = "P_pump"

# Artificial coefficient switches (set True to zero out that term)
ZERO_M = False
ZERO_LAM = False
ZERO_NCOEF = False

# Continuous (off-grid) triad seed (for reproducibility)
TRIAD_SEED = 0

kx0, ky0 = 0.0, 1.0
(qx_t, qy_t, kx_t, ky_t, px_t, py_t) = make_triad(kx=kx0, ky=ky0, y_eps=1e-6, seed=TRIAD_SEED)

# Stored mode ordering: [q, k, p]
qx, qy = float(qx_t), float(qy_t)
kx, ky = float(kx_t), float(ky_t)
px, py = float(px_t), float(py_t)

q2 = qx * qx + qy * qy
k2 = kx * kx + ky * ky
p2 = px * px + py * py

# Iy is just the zonal/nonzonal selector used in the model.
Iy_q = 1.0 if abs(qy) > 0.0 else 0.0
Iy_k = 1.0 if abs(ky) > 0.0 else 0.0
Iy_p = 1.0 if abs(py) > 0.0 else 0.0

Nk = 3

dtshow=0.1
dtstep,dtsavecb=0.02,0.4
t0,t1=0.0,600 #100/gammax #600/gammax
rtol,atol=1e-8,1e-10

output_dir = "data_instability/"
os.makedirs(output_dir, exist_ok=True)
filename = output_dir + f'out_instability_{INIT_CASE}.h5'

#%% Functions

def init_fields(
    case="both_different_phase",
    pump_mode="k",
    pump_amp=1.0,
    seed_amp=1e-6,
    seed=None,
):
    # Initialize only the stored triad modes [q,k,p]
    #
    # Supported cases (exactly four):
    #   - "phi_pump"            : pure potential pump
    #   - "P_pump"              : pure pressure pump
    #   - "both_same_phase"     : both pumped with the same phase
    #   - "both_different_phase": both pumped with different (non-zero) phase difference

    rng = np.random.default_rng(seed)

    mode_to_idx = {"q": 0, "k": 1, "p": 2}
    if pump_mode not in mode_to_idx:
        raise ValueError("pump_mode must be one of 'q', 'k', 'p'")
    pump_idx = mode_to_idx[pump_mode]

    phase_phi = np.exp(1j * 2 * np.pi * rng.random(Nk))
    phase_P = np.exp(1j * 2 * np.pi * rng.random(Nk))

    # Start with small-amplitude seeds on all modes.
    phi0 = (seed_amp * phase_phi).astype(np.complex128)
    P0 = (seed_amp * phase_P).astype(np.complex128)

    theta_phi = float(2 * np.pi * rng.random())
    phi_pump = pump_amp * np.exp(1j * theta_phi)

    if case == "phi_pump":
        phi0[pump_idx] = phi_pump
    elif case == "P_pump":
        theta_P = float(2 * np.pi * rng.random())
        P0[pump_idx] = pump_amp * np.exp(1j * theta_P)
    elif case == "both_same_phase":
        phi0[pump_idx] = phi_pump
        P0[pump_idx] = pump_amp * np.exp(1j * theta_phi)
    elif case == "both_different_phase":
        theta_P = float(2 * np.pi * rng.random())
        dtheta = (theta_P - theta_phi) % (2 * np.pi)
        if dtheta == 0.0:
            theta_P = theta_phi + (np.pi / 2)
        phi0[pump_idx] = phi_pump
        P0[pump_idx] = pump_amp * np.exp(1j * theta_P)
    else:
        raise ValueError(
            "Unknown case. Use one of: "
            "'phi_pump', 'P_pump', 'both_same_phase', 'both_different_phase'"
        )

    Phik = np.asarray(phi0, dtype=np.complex128)
    Pk = np.asarray(P0, dtype=np.complex128)
    return np.hstack((Phik, Pk))

def rhs(t, y):
    # y is complex-valued: [Phi_q, Phi_k, Phi_p, P_q, P_k, P_p]
    # Index convention in this file:
    #   0 -> q-mode, 1 -> k-mode, 2 -> p-mode
    y = np.asarray(y, dtype=np.complex128)
    dy = np.zeros_like(y)
    Phik, Pk = y[:Nk], y[Nk:]
    dPhikdt, dPkdt = dy[:Nk], dy[Nk:]

    phi_star = np.conj(Phik)
    P_star = np.conj(Pk)

    denom_k = float(Iy_k + k2)
    denom_p = float(Iy_p + p2)
    denom_q = float(Iy_q + q2)

    # --- Update k-mode (partners: p, q) ---
    if denom_k != 0.0:
        cross = (px * qy - py * qx)
        M = 0.5 * cross
        Lam = 0.5 * cross * ((Iy_q - Iy_p) + (q2 - p2)) / denom_k
        Ncoef = -0.5 * cross / denom_k

        if ZERO_M:
            M = 0.0
        if ZERO_LAM:
            Lam = 0.0
        if ZERO_NCOEF:
            Ncoef = 0.0

        kdp = (kx * px + ky * py)
        kdq = (kx * qx + ky * qy)

        phi_p_star = phi_star[2]
        phi_q_star = phi_star[0]
        P_p_star = P_star[2]
        P_q_star = P_star[0]

        dPkdt[1] += M * (phi_p_star * P_q_star - phi_q_star * P_p_star)
        dPhikdt[1] += Lam * (phi_p_star * phi_q_star) + Ncoef * (
            kdp * (phi_p_star * P_q_star) - kdq * (phi_q_star * P_p_star)
        )

    # --- Update p-mode (partners: q, k) ---
    if denom_p != 0.0:
        cross = (qx * ky - qy * kx)
        M = 0.5 * cross
        Lam = 0.5 * cross * ((Iy_k - Iy_q) + (k2 - q2)) / denom_p
        Ncoef = -0.5 * cross / denom_p

        if ZERO_M:
            M = 0.0
        if ZERO_LAM:
            Lam = 0.0
        if ZERO_NCOEF:
            Ncoef = 0.0

        pdotq = (px * qx + py * qy)
        pdotk = (px * kx + py * ky)

        phi_q_star = phi_star[0]
        phi_k_star = phi_star[1]
        P_q_star = P_star[0]
        P_k_star = P_star[1]

        dPkdt[2] += M * (phi_q_star * P_k_star - phi_k_star * P_q_star)
        dPhikdt[2] += Lam * (phi_q_star * phi_k_star) + Ncoef * (
            pdotq * (phi_q_star * P_k_star) - pdotk * (phi_k_star * P_q_star)
        )

    # --- Update q-mode (partners: k, p) ---
    if denom_q != 0.0:
        cross = (kx * py - ky * px)
        M = 0.5 * cross
        Lam = 0.5 * cross * ((Iy_p - Iy_k) + (p2 - k2)) / denom_q
        Ncoef = -0.5 * cross / denom_q

        if ZERO_M:
            M = 0.0
        if ZERO_LAM:
            Lam = 0.0
        if ZERO_NCOEF:
            Ncoef = 0.0

        qdotk = (qx * kx + qy * ky)
        qdotp = (qx * px + qy * py)

        phi_k_star = phi_star[1]
        phi_p_star = phi_star[2]
        P_k_star = P_star[1]
        P_p_star = P_star[2]

        dPkdt[0] += M * (phi_k_star * P_p_star - phi_p_star * P_k_star)
        dPhikdt[0] += Lam * (phi_k_star * phi_p_star) + Ncoef * (
            qdotk * (phi_k_star * P_p_star) - qdotp * (phi_p_star * P_k_star)
        )
    return dy

#%% Run the simulation

fl = h5.File(filename, 'w', libver='latest')
fl.swmr_mode = True

zk0 = init_fields(case=INIT_CASE, pump_mode="k", pump_amp=1.0, seed_amp=1e-6, seed=0)
t_eval = np.arange(t0, t1 + 0.5 * dtsavecb, dtsavecb, dtype=float)

save_data(
    fl,
    'data',
    ext_flag=False,
    kx=np.asarray([qx, kx, px], dtype=float),
    ky=np.asarray([qy, ky, py], dtype=float),
    t0=t0,
    t1=t1,
)
save_data(
    fl,
    'params',
    ext_flag=False,
    Npx=Npx,
    Npy=Npy,
    Lx=Lx,
    Ly=Ly,
    INIT_CASE=INIT_CASE,
    TRIAD_SEED=TRIAD_SEED,
    ZERO_M=ZERO_M,
    ZERO_LAM=ZERO_LAM,
    ZERO_NCOEF=ZERO_NCOEF,
)

if t_eval.size == 0:
    fl.close()
    raise SystemExit

sol = solve_ivp(
    rhs,
    t_span=(float(t_eval[0] if t_eval[0] < t0 else t0), float(t1)),
    y0=np.asarray(zk0, dtype=np.complex128),
    method='DOP853',
    t_eval=t_eval,
    rtol=rtol,
    atol=atol,
    max_step=dtstep,
)

if not sol.success:
    fl.close()
    raise RuntimeError(f"solve_ivp failed: {sol.message}")

for ti, yi in zip(sol.t, sol.y.T, strict=False):
    Phik = yi[:Nk]
    Pk = yi[Nk:]
    Omk = np.asarray([-q2 * Phik[0], -k2 * Phik[1], -p2 * Phik[2]], dtype=np.complex128)

    save_data(fl, 'fields', ext_flag=True, Omk=Omk, Pk=Pk, t=float(ti))
    save_data(fl, 'last', ext_flag=False, zk=np.asarray(yi), t=float(ti))

fl.close()
