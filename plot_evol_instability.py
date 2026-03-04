#%% Importing libraries
import os
import h5py as h5
import numpy as np
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'lines.linewidth': 4,
    'axes.linewidth': 3,
    'xtick.major.width': 3,
    'ytick.major.width': 3,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.minor.width': 1.5,
    'ytick.minor.width': 1.5,
    'savefig.dpi': 100,
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.edgecolor': 'black'
})

def mode_label(i: int, kx: float, ky: float) -> str:
    mode_name = ("q", "k", "p")[i] if i < 3 else f"m{i}"
    return rf"$\boldsymbol{{{mode_name}}}$=({kx:.3g},{ky:.3g})"

#%% Load HDF5 file

datadir = "data_instability/"
case = "1_1_1"
pump_mode = 1
delta = 0.5*np.pi
fname = os.path.join(datadir, f"out_instability_{case}_pump_{pump_mode}_delta_{str(round(delta/np.pi,2)).replace('.', '_')}_pi.h5")

with h5.File(fname, "r", swmr=True) as fl:
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

# Compute phase difference arg(phi) - arg(P) for q,k,p
delta = np.unwrap(np.angle(Pk * np.conj(Phik)), axis=1)

# Conserved quantities.
phi_amp2 = np.abs(Phik) ** 2
P_amp2 = np.abs(Pk) ** 2

q2 = qx**2 + qy**2
k2 = kx**2 + ky**2
p2 = px**2 + py**2

consv1 = P_amp2[0, :] + P_amp2[1, :] + P_amp2[2, :]
consv2 = (1.0 + q2) * phi_amp2[0, :] + (1.0 + k2) * phi_amp2[1, :] + (1.0 + p2) * phi_amp2[2, :]
consv3 = (1.0 + q2) * np.abs(Phik[0, :] + Pk[0, :]) ** 2 + (1.0 + k2) * np.abs(Phik[1, :] + Pk[1, :]) ** 2 + (1.0 + p2) * np.abs(Phik[2, :] + Pk[2, :]) ** 2

#%% Plotting

# |Phi_k| vs t
fig = plt.figure(figsize=(16, 9))
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
fig.savefig(fname.replace("out_instability_", "phi_amp_vs_t_").replace(".h5", ".pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# growth rate fit
fig = plt.figure(figsize=(16, 9))
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
fig.savefig(fname.replace("out_instability_", "phi_amp_vs_t_fit_").replace(".h5", ".pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# |P_k| vs t
fig = plt.figure(figsize=(16, 9))
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
fig.savefig(fname.replace("out_instability_", "P_amp_vs_t_").replace(".h5", ".pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# |np.conj(Phi_p)*np.conj(P_q) - np.conj(Phi_q)*np.conj(P_p)| vs t
fig = plt.figure(figsize=(16, 9))
triad_product = np.conj(Phik[2, :]) * np.conj(Pk[0, :]) - np.conj(Phik[0, :]) * np.conj(Pk[2, :])
plt.plot(t, np.abs(triad_product), label="Triad product")
plt.xlabel("t")
plt.ylabel(r"$|\phi_p^* P_q^* - \phi_q^* P_p^*|$")
plt.title(r"Triad interaction: $|\phi_p^* P_q^* - \phi_q^* P_p^*|$ vs time")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(fname.replace("out_instability_", "triad_product_vs_t_").replace(".h5", ".pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# Conserved quantities vs time
fig = plt.figure(figsize=(16, 9))
plt.plot(t, consv1, label=r"$|P_q|^2 + |P_k|^2 + |P_p|^2$")
plt.plot(t, consv2, label=r"$(1+q^2)|\phi_q|^2 + (1+k^2)|\phi_k|^2 + (1+p^2)|\phi_p|^2$")
plt.plot(t, consv3, label=r"$(1+q^2)|\phi_q+P_q|^2 + (1+k^2)|\phi_k+P_k|^2 + (1+p^2)|\phi_p+P_p|^2$")
plt.axhline(0.0, color="k", linestyle="--", linewidth=1.5, alpha=0.7)
plt.xlabel("t")
plt.ylabel("Conserved quantity")
plt.title("Conserved quantities vs time")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(fname.replace("out_instability_", "conserved_quantities_vs_t_").replace(".h5", ".pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# cos(delta) for q,k,p
fig = plt.figure(figsize=(16, 9))
for i in range(min(3, Nk)):
    kx_i, ky_i = modes[i]
    plt.plot(t, np.cos(delta[i, :]), label=mode_label(i, kx_i, ky_i))
plt.xlabel("t")
plt.ylabel(r"$\cos\left(\delta\right)$")
plt.title(r"$\cos\left(\delta\right)$ (q,k,p)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(fname.replace("out_instability_", "cos_delta_vs_t_qkp_").replace(".h5", ".pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)
# %%
