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

ZONALk=False
ZONALp=True
Zonalq=False
pump_mode = 2 # 0: k is smallest, 1: k is medium, 2: k is largest
delta_vals = np.linspace(0, np.pi, 51)  # phase difference between pump modes

cases = [(0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
case_names = ['Pure dia','ITG w.o. diamagnetic','HM-like w. diamagnetic', 'ITG w.o. electrostatic','ITG full']

if ZONALk:
    vec1 = (0.3, 0.0)
    vec2 = (0.0, 0.6)
    vec3 = (-0.3, -0.6)
elif ZONALp:
    vec1 = (0.0, 0.15)
    vec2 = (0.3, 0.0)
    vec3 = (-0.3, -0.15)
elif Zonalq:
    vec1 = (0.0, 0.15)
    vec2 = (0.3, 0.3)
    vec3 = (0.3, 0)
else:
    # (qx, qy, kx, ky, px, py) = make_triad(kx=0.0, ky=0.7, seed=3)
    vec1 = (0.3, 0.3)
    vec2 = (0.0, 0.6)
    vec3 = (-0.3, -0.9)

if pump_mode == 0:
    kx, ky = vec1
    px, py = vec2
    qx, qy = vec3
elif pump_mode == 1:
    qx, qy = vec1
    kx, ky = vec2
    px, py = vec3
elif pump_mode == 2:
    px, py = vec1
    qx, qy = vec2
    kx, ky = vec3

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
    
    # matrix structure: [[0, A], [B, 0]]
    A = np.zeros((len(delta_vals), 2, 2), dtype=complex)
    B = np.zeros((len(delta_vals), 2, 2), dtype=complex)
    
    A[:, 0, 0] = M * Phi0.conj()
    A[:, 0, 1] = -M * P0.conj()
    A[:, 1, 0] = Nqkp * qdk * Phi0.conj()
    A[:, 1, 1] = Lqkp * Phi0.conj() - Nqkp * pdq * P0.conj()
    
    B[:, 0, 0] = -M * Phi0
    B[:, 0, 1] = M * P0
    B[:, 1, 0] = -Npqk * kdp * Phi0
    B[:, 1, 1] = Lpqk * Phi0 + Npqk * pdq * P0
    
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
    plt.axhline(np.abs(Phi0)*np.sqrt(Lqkp*Lpqk),color='k',linestyle='--',label=r"$\lambda=\lambda_{\mathbf{\mathrm{E}}\times\mathbf{\mathrm{B}}}$")
else:
    plt.axhline(0,color='k',linestyle='--',label=r"$\lambda=0$")
plt.xlabel(r'$\left|\delta\right| / \pi$')
plt.ylabel(r'Growth rate $\lambda$')
plt.title(r'$\lambda$ vs $\left|\delta\right|$; pump = ' + pump_labels.get(pump_mode, str(pump_mode)))
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()
if ZONALk:
    plt.savefig(datadir+f"growth_rate_vs_delta_zonalk_pump_mode_{pump_mode}.pdf", bbox_inches="tight")
elif ZONALp:
    plt.savefig(datadir+f"growth_rate_vs_delta_zonalp_pump_mode_{pump_mode}.pdf", bbox_inches="tight")
else:
    plt.savefig(datadir+f"growth_rate_vs_delta_pump_mode_{pump_mode}.pdf", bbox_inches="tight")
plt.show()
# %%
