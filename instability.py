#%% Import libraries

import numpy as np
import h5py as h5
from scipy.integrate import solve_ivp
from modules.gensolver import save_data
from modules.triads import make_triad
import os

#%% Parameters

# "Phi_pump", "P_pump", "both_same_phase", "both_out_of_phase"
INIT_CASE = "both_same_phase"
PUMP_MODE = 1  # 0=q, 1=k, 2=p

# zero when val is 0
ZERO_M = 1
ZERO_LAM = 1
ZERO_N = 0

(qx, qy, kx, ky, px, py) = make_triad(kx=0.0, ky=1.0, seed=0)
q2, k2, p2 = qx**2 + qy**2, kx**2 + ky**2, px**2 + py**2

Nk = 3
dtstep,dtsavecb=0.02,0.4
t0,t1=0.0,520.0
rtol,atol=1e-8,1e-10

output_dir = "data_instability/"
os.makedirs(output_dir, exist_ok=True)
filename = output_dir + f'out_instability_{INIT_CASE}_{ZERO_M}_{ZERO_LAM}_{ZERO_N}.h5'

#%% Functions

def init_fields(pump_mode=1, init_case="both_out_of_phase"):
    # Supported cases:
    #   - "Phi_pump"            : pure potential pump
    #   - "P_pump"              : pure pressure pump
    #   - "both_same_phase"     : both pumped with the same phase
    #   - "both_out_of_phase"   : both pumped pi/2 out of phase
    # pump_mode: 0=q, 1=k, 2=p

    Phi0 = 1e-6 * np.exp(1j * 2 * np.pi * np.random.random(Nk))
    P0 = 1e-6 * np.exp(1j * 2 * np.pi * np.random.random(Nk))
    pump = np.exp(1j * 2 * np.pi * np.random.random())

    if init_case == "Phi_pump":
        Phi0[pump_mode] = pump
    elif init_case == "P_pump":
        P0[pump_mode] = pump
    elif init_case == "both_same_phase":
        Phi0[pump_mode] = pump
        P0[pump_mode] = pump
    elif init_case == "both_out_of_phase":
        Phi0[pump_mode] = pump
        P0[pump_mode] = 1j * pump
    else:
        raise ValueError("Unknown case")
    
    return np.hstack((Phi0, P0))

def rhs(t, y):
    print(f"t = {t}")
    dy = np.zeros_like(y)
    Phik, Pk = y[:Nk], y[Nk:]
    dPhikdt, dPkdt = dy[:Nk], dy[Nk:]

    sigq, sigk, sigp = np.sign(qy), np.sign(ky), np.sign(py)
    cross = (px * qy - py * qx)
    M = ZERO_M * 0.5 * cross
    kdp = (kx * px + ky * py)
    qdk = (kx * qx + ky * qy)
    pdq = (px * qx + py * qy)

    # --- Update q-mode ---
    Lam = ZERO_LAM * 0.5 * cross * ((sigp - sigk) + (p2 - k2)) / (sigq + q2)
    N = ZERO_N * -0.5 * cross / (sigq + q2)

    dPkdt[0] += M * (np.conj(Phik[1]) * np.conj(Pk[2]) - np.conj(Phik[2]) * np.conj(Pk[1]))
    dPhikdt[0] += Lam * (np.conj(Phik[1]) * np.conj(Phik[2])) + N * (qdk * (np.conj(Phik[1]) * np.conj(Pk[2])) - pdq * (np.conj(Phik[2]) * np.conj(Pk[1])))
                                                                     
    # --- Update k-mode ---
    Lam = ZERO_LAM * 0.5 * cross * ((sigq - sigp) + (q2 - p2)) / (sigk + k2)
    N = ZERO_N * -0.5 * cross / (sigk + k2)

    dPkdt[1] += M * (np.conj(Phik[2]) * np.conj(Pk[0]) - np.conj(Phik[0]) * np.conj(Pk[2]))
    dPhikdt[1] += Lam * (np.conj(Phik[2]) * np.conj(Phik[0])) + N * (kdp * (np.conj(Phik[2]) * np.conj(Pk[0])) - qdk * (np.conj(Phik[0]) * np.conj(Pk[2])))

    # --- Update p-mode ---
    Lam = ZERO_LAM * 0.5 * cross * ((sigk - sigq) + (k2 - q2)) / (sigp + p2)
    N = ZERO_N * -0.5 * cross / (sigp + p2)

    dPkdt[2] += M * (np.conj(Phik[0]) * np.conj(Pk[1]) - np.conj(Phik[1]) * np.conj(Pk[0]))
    dPhikdt[2] += Lam * (np.conj(Phik[0]) * np.conj(Phik[1])) + N * (pdq * (np.conj(Phik[0]) * np.conj(Pk[1])) - kdp * (np.conj(Phik[1]) * np.conj(Pk[0])))
    return dy

#%% Run the simulation

fl = h5.File(filename, 'w', libver='latest')
fl.swmr_mode = True

save_data(fl, 'data', ext_flag=False, qx=qx, kx=kx, px=px, qy=qy, ky=ky, py=py, t0=t0, t1=t1)
zk0 = init_fields(pump_mode=PUMP_MODE, init_case=INIT_CASE)
t_eval = np.arange(t0, t1, dtsavecb)
sol = solve_ivp(rhs, t_span=(t0, t1),y0=zk0,method='DOP853',t_eval=t_eval,rtol=rtol,atol=atol,max_step=dtstep)
if not sol.success:
    fl.close()
    raise RuntimeError(f"solve_ivp failed: {sol.message}")

fl['t'] = sol.t
fl['Phik'] = sol.y[:Nk, :]
fl['Pk'] = sol.y[Nk:, :]

fl.close()