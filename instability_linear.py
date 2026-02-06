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

pump_mode = 1  # 0=q, 1=k, 2=p
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

datadir = "data_instability/"
os.makedirs(datadir, exist_ok=True)

Phi0 = np.exp(1j * 2 * np.pi * np.random.random())

#%% Compute growth rate

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

    if pump_mode==0:
        tr = np.abs(Phi0)**2*(Lkpq*Lpqk - M**2) - np.abs(P0)**2*Nkpq*Npqk*kdp**2 + Phi0*P0.conj()*(M*Npqk*pdq + Nkpq*Lpqk*kdp) + Phi0.conj()*P0*(Nkpq*M*qdk - Lkpq*Npqk*kdp)
        det = -M**2*Phi0**2*(Lkpq*Phi0.conj() + Nkpq*P0.conj()*(kdp-qdk))*(Lpqk*Phi0 - Npqk*P0*(kdp-pdq))
    elif pump_mode==1:
        tr = np.abs(Phi0)**2*(Lqkp*Lpqk - M**2) - np.abs(P0)**2*Nqkp*Npqk*pdq**2 + Phi0*P0.conj()*(M*Npqk*kdp - Nqkp*Lpqk*pdq) + Phi0.conj()*P0*(Nqkp*M*qdk + Lqkp*Npqk*pdq)
        det = -M**2*Phi0**2*(Lqkp*Phi0.conj() + Nqkp*P0.conj()*(qdk - pdq))*(Lpqk*Phi0 - Npqk*P0*(kdp - pdq))
    elif pump_mode==2:
        tr = np.abs(Phi0)**2*(Lqkp*Lkpq - M**2) - np.abs(P0)**2*Nqkp*Nkpq*qdk**2 + Phi0*P0.conj()*(M*Nkpq*pdq + Nqkp*Lkpq*qdk) + Phi0.conj()*P0*(Nqkp*M*kdp - Lqkp*Nkpq*qdk)
        det = -M**2*Phi0**2*(Lqkp*Phi0.conj() + Nqkp*P0.conj()*(pdq - qdk))*(Lkpq*Phi0 - Nkpq*P0*(kdp - qdk))

    lam1 = np.real(np.sqrt(tr + np.sqrt(tr**2 - 4 * det)) / np.sqrt(2))
    lam2 = np.real(np.sqrt(tr - np.sqrt(tr**2 - 4 * det)) / np.sqrt(2))
    lam_vals[case_idx, :] = np.maximum.reduce([lam1, lam2, -lam1, -lam2])

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
plt.title(r'$\lambda$ vs $\delta$')
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(datadir+f"growth_rate_vs_delta_pump_mode_{pump_mode}.pdf", bbox_inches="tight")
plt.show()