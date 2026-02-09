#%% Importing libraries
import os
import h5py as h5
import numpy as np
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

def mode_label(i: int, kx: float, ky: float) -> str:
    mode_name = ("q", "k", "p")[i] if i < 3 else f"m{i}"
    return rf"$\boldsymbol{{{mode_name}}}$=({kx:.3g},{ky:.3g})"

#%% Load HDF5 file

datadir = "data_instability/"
case = "1_1_0"
pump_mode = 1
delta = 0.5*np.pi
file_name = os.path.join(datadir, f"out_instability_{case}_pump_{pump_mode}_delta_{str(round(delta/np.pi,2)).replace('.', '_')}_pi.h5")

with h5.File(file_name, "r", swmr=True) as fl:
    t = np.asarray(fl["t"][:], dtype=float)
    Phik = np.asarray(fl["Phik"][:])
    Pk = np.asarray(fl["Pk"][:])
    qx = float(fl["data/qx"][()])
    qy = float(fl["data/qy"][()])
    kx = float(fl["data/kx"][()])
    ky = float(fl["data/ky"][()])
    px = float(fl["data/px"][()])
    py = float(fl["data/py"][()])
    M = float(fl["data/M"][()])
    Lqkp = float(fl["data/Lqkp"][()])
    Lkpq = float(fl["data/Lkpq"][()])
    Lpqk = float(fl["data/Lpqk"][()])
    Nqkp = float(fl["data/Nqkp"][()])
    Nkpq = float(fl["data/Nkpq"][()])
    Npqk = float(fl["data/Npqk"][()])

nt=len(t)

Phi0 = Phik[pump_mode, 0]
if pump_mode == 1:
    gam = np.abs(Phi0)*np.sqrt(Lqkp*Lpqk)
    gam_exp = r"$\left|\phi_0\right|\Lambda^q_{kp}\Lambda^p_{qk}$"
else:
    gam = 0
    gam_exp = "$0$"
    
# Store wavenumbers by mode
modes = [(qx, qy), (kx, ky), (px, py)]
Nk = Phik.shape[0]

phase_diff = np.unwrap(np.angle(Phik * np.conj(Pk)), axis=1)

#%% Plotting

# |Phi_k| vs t
fig = plt.figure(figsize=(16,9))
for i in range(Nk):
    kx_i, ky_i = modes[i]
    plt.plot(t, np.abs(Phik)[i, :], label=mode_label(i, kx_i, ky_i))
plt.xlabel("t")
plt.ylabel(r"$|\phi_k|$")
plt.title(r"Triad amplitudes: $|\phi_k|$ vs time")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(file_name.replace("out_instability_", "phi_amp_vs_t_").replace(".h5", ".pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# growth rate fit
fig = plt.figure(figsize=(16,9))
for i in [0,2]:
    kx_i, ky_i = modes[i]
    plt.semilogy(t[:int(nt/2)], np.abs(Phik)[i, :int(nt/2)], label=mode_label(i, kx_i, ky_i))
plt.semilogy(t[:int(nt/2)], 1e-6 * np.exp(gam*t[:int(nt/2)]), 'k--', label=r'$\gamma$ = '+gam_exp)
plt.xlabel("t")
plt.ylabel(r"$|\phi_k|$")
plt.title(r"Triad amplitudes: $|\phi_k|$ vs time (fit)")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(file_name.replace("out_instability_", "phi_amp_vs_t_fit_").replace(".h5", ".pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# |P_k| vs t
fig = plt.figure(figsize=(16,9))
for i in range(Nk):
    kx_i, ky_i = modes[i]
    plt.plot(t, np.abs(Pk)[i, :], label=mode_label(i, kx_i, ky_i))
plt.xlabel("t")
plt.ylabel(r"$|P_k|$")
plt.title(r"Triad amplitudes: $|P_k|$ vs time")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(file_name.replace("out_instability_", "P_amp_vs_t_").replace(".h5", ".pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# |np.conj(Phi_p)*np.conj(P_q) - np.conj(Phi_q)*np.conj(P_p)| vs t
fig = plt.figure(figsize=(16,9))
triad_product = np.conj(Phik[2, :]) * np.conj(Pk[0, :]) - np.conj(Phik[0, :]) * np.conj(Pk[2, :])
plt.plot(t, np.abs(triad_product), label="Triad product")
plt.xlabel("t")
plt.ylabel(r"$|\phi_p^* P_q^* - \phi_q^* P_p^*|$")
plt.title(r"Triad interaction: $|\phi_p^* P_q^* - \phi_q^* P_p^*|$ vs time")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(file_name.replace("out_instability_", "triad_product_vs_t_").replace(".h5", ".pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# phase difference arg(phi) - arg(P) for q,k,p
fig = plt.figure(figsize=(16,9))
for i in range(min(3, Nk)):
    kx_i, ky_i = modes[i]
    plt.plot(t, phase_diff[i,:], label=mode_label(i, kx_i, ky_i))
plt.xlabel("t")
plt.ylabel(r"$\arg(\phi_k) - \arg(P_k)$")
plt.title(r"Phase difference: $\arg(\phi_k) - \arg(P_k)$ (q,k,p)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(file_name.replace("out_instability_", "phi_minus_P_phase_vs_t_qkp_").replace(".h5", ".pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)
# %%
