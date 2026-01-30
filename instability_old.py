#%% Import libraries

import numpy as np
import cupy as cp
import h5py as h5
from modules.gensolver import Gensolver,save_data
from functools import partial
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
kx = cp.asarray([qx_t, kx_t, px_t])
ky = cp.asarray([qy_t, ky_t, py_t])
kpsq = kx**2 + ky**2
Nk = int(kx.size)

dtshow=0.1
dtstep,dtsavecb=0.02,0.4
t0,t1=0.0,600 #100/gammax #600/gammax
rtol,atol=1e-8,1e-10
wecontinue=True

output_dir = "data_instability/"
os.makedirs(output_dir, exist_ok=True)
filename = output_dir + f'out_instability_{INIT_CASE}.h5'
if not os.path.exists(filename):
    wecontinue=False

#%% Functions

_kxv = cp.asnumpy(kx).astype(float)
_kyv = cp.asnumpy(ky).astype(float)
_k2v = _kxv**2 + _kyv**2
_Iy = (np.abs(_kyv) > 0.0).astype(float)

# Precompute all triad interactions using conjugate symmetry.
# We sum over p,q in {±q, ±k, ±p} such that p+q+k=0 (within tolerance),
# but only evolve k in {q,k,p}.
CLOSURE_TOL = 1e-12

# Mode representation: (idx in 0..2, sign in {+1,-1}, kx, ky)
_all_modes: list[tuple[int, int, float, float]] = []
for i in range(Nk):
    _all_modes.append((i, +1, float(_kxv[i]), float(_kyv[i])))
    _all_modes.append((i, -1, float(-_kxv[i]), float(-_kyv[i])))

def _amp_pos(x_hat: cp.ndarray, idx: int, sign: int) -> cp.ndarray:
    # Returns X(sign * k_idx), assuming X(-k)=conj(X(k)).
    return x_hat[idx] if sign == +1 else cp.conj(x_hat[idx])

def _amp_star(x_hat: cp.ndarray, idx: int, sign: int) -> cp.ndarray:
    # Returns X_k^* for the mode (sign * k_idx).
    return cp.conj(_amp_pos(x_hat, idx, sign))

# Precompute terms per target k (only + modes: q,k,p)
_triad_terms: list[list[tuple[int, int, int, int, float, float, float, float, float]]] = [[] for _ in range(Nk)]
# tuple: (p_idx, p_sign, q_idx, q_sign, M, Lambda, N, k_dot_p, k_dot_q)
for k_idx in range(Nk):
    kx_, ky_ = float(_kxv[k_idx]), float(_kyv[k_idx])
    denom = (_Iy[k_idx] + _k2v[k_idx])
    if denom == 0.0:
        continue

    for p_idx, p_sign, px, py in _all_modes:
        for q_idx, q_sign, qx, qy in _all_modes:
            if (abs(px + qx + kx_) > CLOSURE_TOL) or (abs(py + qy + ky_) > CLOSURE_TOL):
                continue

            cross = (px * qy - py * qx)
            if cross == 0.0:
                continue

            M = 0.5 * cross
            Lambda = 0.5 * cross * (_Iy[q_idx] - _Iy[p_idx] + _k2v[q_idx] - _k2v[p_idx]) / denom
            Ncoef = -0.5 * cross / denom

            if ZERO_M:
                M = 0.0
            if ZERO_LAM:
                Lambda = 0.0
            if ZERO_NCOEF:
                Ncoef = 0.0
            if (M == 0.0) and (Lambda == 0.0) and (Ncoef == 0.0):
                continue

            k_dot_p = (kx_ * px + ky_ * py)
            k_dot_q = (kx_ * qx + ky_ * qy)

            _triad_terms[k_idx].append(
                (p_idx, p_sign, q_idx, q_sign, float(M), float(Lambda), float(Ncoef), float(k_dot_p), float(k_dot_q))
            )

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

    Phik = cp.asarray(phi0, dtype=cp.complex128)
    Pk = cp.asarray(P0, dtype=cp.complex128)
    return cp.hstack((Phik, Pk))

def fsavecb(t,y,flag):
    zk=y.view(dtype=complex)
    Phik,Pk=zk[:Nk],zk[Nk:]
    Omk=-kpsq*Phik
    if flag=='fields':
        save_data(fl,'fields',ext_flag=True,Omk=Omk.get(),Pk=Pk.get(),t=t)
    elif flag=='zonal':
        pass
    elif flag=='fluxes':
        pass
    save_data(fl,'last',ext_flag=False,zk=zk.get(),t=t)

def fshowcb(t,y):
    zk=y.view(dtype=complex)
    Phik,Pk=zk[:Nk],zk[Nk:]
    Ktot = cp.sum(kpsq * cp.abs(Phik)**2)
    print(f'Ktot={float(Ktot.get()):.3g}')

def rhs_itg(t,y):
    zk=y.view(dtype=complex)
    dzkdt=cp.zeros_like(zk)
    Phik,Pk=zk[:Nk],zk[Nk:]
    dPhikdt,dPkdt=dzkdt[:Nk],dzkdt[Nk:]

    for k_idx in range(Nk):
        for p_idx, p_sign, q_idx, q_sign, M, Lam, Ncoef, kdp, kdq in _triad_terms[k_idx]:
            phi_p_star = _amp_star(Phik, p_idx, p_sign)
            phi_q_star = _amp_star(Phik, q_idx, q_sign)
            P_p_star = _amp_star(Pk, p_idx, p_sign)
            P_q_star = _amp_star(Pk, q_idx, q_sign)

            dPkdt[k_idx] += M * (phi_p_star * P_q_star - phi_q_star * P_p_star)
            dPhikdt[k_idx] += Lam * (phi_p_star * phi_q_star) + Ncoef * (
                kdp * (phi_p_star * P_q_star) - kdq * (phi_q_star * P_p_star)
            )
    return dzkdt.view(dtype=float)

#%% Run the simulation

if(wecontinue):
    fl=h5.File(filename,'r+',libver='latest')
    fl.swmr_mode = True
    zk=fl['last/zk'][()]
    t0=fl['last/t'][()]
else:
    fl=h5.File(filename,'w',libver='latest')
    fl.swmr_mode = True
    zk=init_fields(case=INIT_CASE, pump_mode="k", pump_amp=1.0, seed_amp=1e-6, seed=0)
    save_data(fl,'data',ext_flag=False,kx=kx.get(),ky=ky.get(),t0=t0,t1=t1)
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

fsave = [partial(fsavecb,flag='fields'), partial(fsavecb,flag='zonal'), partial(fsavecb,flag='fluxes')]
dtsave=[10*dtsavecb,dtsavecb,dtsavecb]
r=Gensolver('cupy_ivp.DOP853',rhs_itg,t0,zk.view(dtype=float),t1,fsave=fsave,fshow=fshowcb,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,dense=False,rtol=rtol,atol=atol)
r.run()
fl.close()

# %%
