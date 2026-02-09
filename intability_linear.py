#%% Import libraries

import numpy as np
import h5py as h5
from scipy.integrate import solve_ivp
from modules.gensolver import save_data
from modules.triads import make_triad
import matplotlib.pyplot as plt
import os

plt.rcParams.update(
    {
        "lines.linewidth": 3,
        "axes.linewidth": 2,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "savefig.dpi": 120,
        "font.size": 22,          # default text
        "axes.titlesize": 30,     # figure title
        "axes.labelsize": 26,     # x/y labels
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 22
    }
)

#%% Parameters

pump_mode = 2  # 0=q, 1=k, 2=p
delta_vals = np.linspace(0.0, 2*np.pi, 101)  # phase difference between pump modes

cases = [(0, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
case_names = ['HM-like w.o. diamagnetic','HM-like full', 'ITG w.o. electrostatic','ITG w.o. diamagnetic','ITG full']

# (qx, qy, kx, ky, px, py) = make_triad(kx=0.0, ky=1.0, seed=0)
qx,qy,kx,ky,px,py=0.4,0.4,0.0,1.0,-0.4,-1.4 
# qx,qy,kx,ky,px,py=1.0,0.0,0.4,0.4,-1.4,-0.4 #q-zonal
q2, k2, p2 = qx**2 + qy**2, kx**2 + ky**2, px**2 + py**2
cross = (px * qy - py * qx)
sigq, sigk, sigp = np.sign(np.abs(qy)), np.sign(np.abs(ky)), np.sign(np.abs(py))
Lqkp_base = cross * (p2 - k2) / (sigq + q2)
Lkpq_base = cross * (q2 - p2) / (sigk + k2)
Lpqk_base = cross * (k2 - q2) / (sigp + p2)
Nqkp_base = -0.5 * cross / (sigq + q2)
Nkpq_base = -0.5 * cross / (sigk + k2)
Npqk_base = -0.5 * cross / (sigp + p2)
kdp = (kx * px + ky * py)
qdk = (kx * qx + ky * qy)
pdq = (px * qx + py * qy)

print(f"M={0.5 * cross}, Lqkp={Lqkp_base}, Lkpq={Lkpq_base}, Lpqk={Lpqk_base}, Nqkp={Nqkp_base}, Nkpq={Nkpq_base}, Npqk={Npqk_base}")

datadir = "data_instability/"
os.makedirs(datadir, exist_ok=True)

Phi0 = np.exp(1j * 2 * np.pi * np.random.random())

#%% Compute growth rate
tr_cases = []
det_cases = []
lam_vals = np.zeros((len(cases), len(delta_vals)))
for case_idx, (zero_m, zero_l, zero_n) in enumerate(cases):
    M = zero_m * 0.5 * cross
    Lqkp = zero_l * Lqkp_base
    Lkpq = zero_l * Lkpq_base
    Lpqk = zero_l * Lpqk_base
    Nqkp = zero_n * Nqkp_base
    Nkpq = zero_n * Nkpq_base
    Npqk = zero_n * Npqk_base

    P0 = np.exp(1j * delta_vals) * Phi0
    if pump_mode == 0:
        mat = np.zeros((len(delta_vals), 4, 4), dtype=complex)
        mat[:, 0, 2] = -M * Phi0.conj()
        mat[:, 0, 3] = M * P0.conj()
        mat[:, 1, 2] = -Nkpq * qdk * Phi0.conj()
        mat[:, 1, 3] = Lkpq * Phi0.conj() + Nkpq * pdq * P0.conj()
        mat[:, 2, 0] = M * Phi0
        mat[:, 2, 1] = -M * P0
        mat[:, 3, 0] = Npqk * kdp * Phi0
        mat[:, 3, 1] = Lpqk * Phi0 - Npqk * pdq * P0

    elif pump_mode == 1:
        mat = np.zeros((len(delta_vals), 4, 4), dtype=complex)
        mat[:, 0, 2] = M * Phi0.conj()
        mat[:, 0, 3] = -M * P0.conj()
        mat[:, 1, 2] = -Nqkp * qdk * Phi0.conj()
        mat[:, 1, 3] = Lqkp * Phi0.conj() - Nqkp * pdq * P0.conj()
        mat[:, 2, 0] = -M * Phi0
        mat[:, 2, 1] = M * P0
        mat[:, 3, 0] = -Npqk * kdp * Phi0
        mat[:, 3, 1] = Lpqk * Phi0 + Npqk * pdq * P0

    elif pump_mode == 2:
        mat = np.zeros((len(delta_vals), 4, 4), dtype=complex)
        mat[:, 0, 2] = M * Phi0.conj()
        mat[:, 0, 3] = -M * P0.conj()
        mat[:, 1, 2] = -Nqkp * qdk * Phi0.conj()
        mat[:, 1, 3] = Lqkp * Phi0.conj() - Nqkp * pdq * P0.conj()
        mat[:, 2, 0] = -M * Phi0
        mat[:, 2, 1] = M * P0
        mat[:, 3, 0] = -Nkpq * qdk * Phi0
        mat[:, 3, 1] = Lkpq * Phi0 + Nkpq * pdq * P0

    eigvals = np.linalg.eigvals(mat)
    lam_vals[case_idx, :] = np.max(np.real(eigvals), axis=1)


#%% Plot

plt.figure(figsize=(16,9))
for i, (zero_m, zero_l, zero_n) in enumerate(cases):
    m_label = r"$\mathrm{M}^k_{pq}=0$" if zero_m == 0 else r"$\mathrm{M}^k_{pq}\neq 0$"
    l_label = r"$\Lambda^k_{pq}=0$" if zero_l == 0 else r"$\Lambda^k_{pq}\neq 0$"
    n_label = r"$\mathrm{N}^k_{pq}=0$" if zero_n == 0 else r"$\mathrm{N}^k_{pq}\neq 0$"
    plt.plot(
        delta_vals / np.pi,
        lam_vals[i],
        'o-',
        label=f"{m_label}, {l_label}, {n_label}: {case_names[i]}",
    )
plt.xlabel(r'$\delta / \pi$')
plt.ylabel(r'Growth rate $\lambda$')
plt.title(r'$\lambda$ vs $\delta$; pump = '+str(pump_mode))
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(datadir+f"growth_rate_vs_delta_pump_mode_{pump_mode}.pdf", bbox_inches="tight")
plt.show()
# %%
