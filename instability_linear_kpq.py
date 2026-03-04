#%% Import libraries

import numpy as np
import matplotlib.pyplot as plt
import os
from modules.triads import make_triad

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

ZONAL=False
pump_mode = 2  # 0=k, 1=p, 2=q
delta_vals = np.linspace(-np.pi, np.pi, 101)  # phase difference between pump modes

cases = [(0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
case_names = ['Pure dia','ITG w.o. diamagnetic','HM-like w. diamagnetic', 'ITG w.o. electrostatic','ITG full']

if ZONAL:
    kx,ky,px,py,qx,qy=0.4,0.0,0.0,0.7,-0.4,-0.7 #k-zonal
else:
    # (qx, qy, kx, ky, px, py) = make_triad(kx=0.0, ky=0.7, seed=3)
    kx,ky,px,py,qx,qy=0.3,0.3,0.0,0.7,-0.3,-1.0

k2, p2, q2 = kx**2 + ky**2, px**2 + py**2, qx**2 + qy**2
cross = (px * qy - py * qx)
sigk, sigp, sigq = np.sign(np.abs(ky)), np.sign(np.abs(py)), np.sign(np.abs(qy))
Lkpq_base = cross * (q2 - p2) / (sigk + k2)
Lpqk_base = cross * (k2 - q2) / (sigp + p2)
Lqkp_base = cross * (p2 - k2) / (sigq + q2)
Nkpq_base = -0.5 * cross / (sigk + k2)
Npqk_base = -0.5 * cross / (sigp + p2)
Nqkp_base = -0.5 * cross / (sigq + q2)
kdp = (kx * px + ky * py)
pdq = (px * qx + py * qy) 
qdk = (kx * qx + ky * qy)

datadir = "data_instability/"
os.makedirs(datadir, exist_ok=True)

Phi0 = np.exp(1j * 2 * np.pi * np.random.random())

#%% Compute growth rate

tr_cases = []
det_cases = []
lam_vals = np.zeros((len(cases), len(delta_vals)))
for case_idx, (zero_m, zero_l, zero_n) in enumerate(cases):
    M = zero_m * 0.5 * cross
    Lkpq = zero_l * Lkpq_base
    Lpqk = zero_l * Lpqk_base
    Lqkp = zero_l * Lqkp_base

    Nkpq = zero_n * Nkpq_base
    Npqk = zero_n * Npqk_base
    Nqkp = zero_n * Nqkp_base

    P0 = np.exp(1j * delta_vals) * Phi0
    
    # Define 2x2 blocks A and B directly (mat structure: [[0, A], [B, 0]])
    A = np.zeros((len(delta_vals), 2, 2), dtype=complex)
    B = np.zeros((len(delta_vals), 2, 2), dtype=complex)

    if pump_mode == 0:
        A[:, 0, 0] = -M * Phi0.conj()
        A[:, 0, 1] = M * P0.conj()
        A[:, 1, 0] = -Npqk * kdp * Phi0.conj()
        A[:, 1, 1] = Lpqk * Phi0.conj() + Npqk * pdq * P0.conj()

        B[:, 0, 0] = M * Phi0
        B[:, 0, 1] = -M * P0
        B[:, 1, 0] = Nqkp * qdk * Phi0
        B[:, 1, 1] = Lqkp * Phi0 - Nqkp * pdq * P0

    elif pump_mode == 1:
        A[:, 0, 0] = M * Phi0.conj()
        A[:, 0, 1] = -M * P0.conj()
        A[:, 1, 0] = Nkpq * kdp * Phi0.conj()
        A[:, 1, 1] = Lkpq * Phi0.conj() - Nkpq * qdk * P0.conj()

        B[:, 0, 0] = -M * Phi0
        B[:, 0, 1] = M * P0
        B[:, 1, 0] = -Nqkp * pdq * Phi0
        B[:, 1, 1] = Lqkp * Phi0 + Nqkp * qdk * P0
    
    elif pump_mode == 2:
        A[:, 0, 0] = -M * Phi0.conj()
        A[:, 0, 1] = M * P0.conj()
        A[:, 1, 0] = -Nkpq * qdk * Phi0.conj()
        A[:, 1, 1] = Lkpq * Phi0.conj() + Nkpq * kdp * P0.conj()
        
        B[:, 0, 0] = M * Phi0
        B[:, 0, 1] = -M * P0
        B[:, 1, 0] = Npqk * pdq * Phi0
        B[:, 1, 1] = Lpqk * Phi0 - Npqk * kdp * P0
    
    AB = A @ B
    lambda2 = np.linalg.eigvals(AB) #(n_delta, 2)
    lam_all = np.concatenate([np.sqrt(lambda2), -np.sqrt(lambda2)], axis=1) #(n_delta, 4)
    lam_vals[case_idx, :] = np.max(np.real(lam_all), axis=1)

#%% Plot

pump_labels = {0: "k", 1: "p", 2: "q"}
roman_labels = ["i", "ii", "iii", "iv", "v"]

plt.figure(figsize=(16, 9))
for i in range(len(cases)):
    plt.plot(delta_vals / np.pi,lam_vals[i],'o-',label=f"{roman_labels[i]}: {case_names[i]}")
if pump_mode == 1:
    plt.axhline(np.abs(Phi0)*np.sqrt(Lkpq*Lqkp),color='k',linestyle='--',label=r"$\lambda=\lambda_{\mathbf{\mathrm{E}}\times\mathbf{\mathrm{B}}}$")
else:
    plt.axhline(0,color='k',linestyle='--',label=r"$\lambda=0$")
plt.xlabel(r'$\delta / \pi$')
plt.ylabel(r'Growth rate $\lambda$')
plt.title(r'$\lambda$ vs $\delta$; pump = ' + pump_labels.get(pump_mode, str(pump_mode)))
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()
if ZONAL:
    plt.savefig(datadir+f"growth_rate_vs_delta_zonalk_pump_mode_{pump_mode}.pdf", bbox_inches="tight")
else:
    plt.savefig(datadir+f"growth_rate_vs_delta_pump_mode_{pump_mode}.pdf", bbox_inches="tight")
plt.show()
# %%
