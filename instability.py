#%% Import libraries

import numpy as np
import h5py as h5
from scipy.integrate import solve_ivp
from modules.gensolver import save_data
from modules.triads import make_triad
import os

#%% Parameters

PUMP_MODE = 1  # 0=q, 1=k, 2=p
delta = 0.5*np.pi  # phase difference between pump modes

# zero when val is 0
ZERO_M = 0
ZERO_L = 1
ZERO_N = 0

Nk = 3
# (qx, qy, kx, ky, px, py) = make_triad(kx=0.0, ky=1.0, seed=0)
qx,qy,kx,ky,px,py=0.4,0.4,0.0,1.0,-0.4,-1.4 
# qx,qy,kx,ky,px,py=1.0,0.0,0.4,0.4,-1.4,-0.4 #q-zonal
q2, k2, p2 = qx**2 + qy**2, kx**2 + ky**2, px**2 + py**2
cross = (px * qy - py * qx)
M = ZERO_M * 0.5 * cross
sigq, sigk, sigp = np.sign(np.abs(qy)), np.sign(np.abs(ky)), np.sign(np.abs(py))
Lqkp = ZERO_L * cross * (p2 - k2) / (sigq + q2)
Lkpq = ZERO_L * cross * (q2 - p2) / (sigk + k2)
Lpqk = ZERO_L * cross * (k2 - q2) / (sigp + p2)
Nqkp = ZERO_N * -0.5 * cross / (sigq + q2)
Nkpq = ZERO_N * -0.5 * cross / (sigk + k2)
Npqk = ZERO_N * -0.5 * cross / (sigp + p2)
kdp = (kx * px + ky * py)
qdk = (kx * qx + ky * qy)
pdq = (px * qx + py * qy)
dtstep,dtsavecb=0.02,0.4
t0,t1=0.0,100.0

output_dir = "data_instability/"
os.makedirs(output_dir, exist_ok=True)
filename = output_dir + f'out_instability_{ZERO_M}_{ZERO_L}_{ZERO_N}_pump_{PUMP_MODE}_delta_{str(round(delta/np.pi,2)).replace(".", "_")}_pi.h5'

#%% Functions

def init_fields(pump_mode=1, delta=np.pi / 2):
    Phi0 = 1e-6 * np.exp(1j * 2 * np.pi * np.random.random(Nk))
    P0 = 1e-6 * np.exp(1j * 2 * np.pi * np.random.random(Nk))
    pump = np.exp(1j * 2 * np.pi * np.random.random())

    Phi0[pump_mode] = pump
    P0[pump_mode] = np.exp(1j * delta) * pump
    
    return np.hstack((Phi0, P0))

def rhs(t, y):
    print(f"t = {t}")
    dy = np.zeros_like(y)
    Phik, Pk = y[:Nk], y[Nk:]
    dPhikdt, dPkdt = dy[:Nk], dy[Nk:]

    dPkdt[0] += M * (Phik[1].conj() * Pk[2].conj() - Phik[2].conj() * Pk[1].conj())
    dPhikdt[0] += Lqkp * (Phik[1].conj() * Phik[2].conj()) + Nqkp * (qdk * (Phik[1].conj() * Pk[2].conj()) - pdq * (Phik[2].conj() * Pk[1].conj()))                                                   
    dPkdt[1] += M * (Phik[2].conj() * Pk[0].conj() - Phik[0].conj() * Pk[2].conj())
    dPhikdt[1] += Lkpq * (Phik[2].conj() * Phik[0].conj()) + Nkpq * (kdp * (Phik[2].conj() * Pk[0].conj()) - qdk * (Phik[0].conj() * Pk[2].conj()))
    dPkdt[2] += M * (Phik[0].conj() * Pk[1].conj() - Phik[1].conj() * Pk[0].conj())
    dPhikdt[2] += Lpqk * (Phik[0].conj() * Phik[1].conj()) + Npqk * (pdq * (Phik[0].conj() * Pk[1].conj()) - kdp * (Phik[1].conj() * Pk[0].conj()))
    return dy

#%% Run the simulation

fl = h5.File(filename, 'w', libver='latest')
fl.swmr_mode = True

save_data(fl, 'data', ext_flag=False, qx=qx, kx=kx, px=px, qy=qy, ky=ky, py=py,M=M, Lqkp=Lqkp, Lkpq=Lkpq, Lpqk=Lpqk, Nqkp=Nqkp, Nkpq=Nkpq, Npqk=Npqk, t0=t0, t1=t1)
zk0 = init_fields(pump_mode=PUMP_MODE, delta=delta)
t_eval = np.arange(t0, t1, dtsavecb)
sol = solve_ivp(rhs, t_span=(t0, t1),y0=zk0,method='DOP853',t_eval=t_eval,rtol=1e-8,atol=1e-10,max_step=dtstep)
if not sol.success:
    fl.close()
    raise RuntimeError(f"solve_ivp failed: {sol.message}")

fl['t'] = sol.t
fl['Phik'] = sol.y[:Nk, :]
fl['Pk'] = sol.y[Nk:, :]

fl.close()