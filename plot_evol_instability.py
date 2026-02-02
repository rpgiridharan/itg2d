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
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    }
)

def _mode_label(i: int, kx: float, ky: float) -> str:
    mode_name = ("q", "k", "p")[i] if i < 3 else f"m{i}"
    return rf"$\boldsymbol{{{mode_name}}}$=({kx:.3g},{ky:.3g})"

#%% Load HDF5 file

datadir = "data_instability/"
# case = "Phi_pump_1_1_1"
case = "both_out_1_1_0"
file_name = os.path.join(datadir, f"out_instability_{case}.h5")

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

# Store wavenumbers by mode
modes = [(qx, qy), (kx, ky), (px, py)]
Nk = Phik.shape[0]

amp_phi = np.abs(Phik)
amp_P = np.abs(Pk)

phase_phi = np.unwrap(np.angle(Phik), axis=1)
phase_P = np.unwrap(np.angle(Pk), axis=1)
phase_diff = np.unwrap(np.angle(Phik * np.conj(Pk)), axis=1)

#%% Plotting

# |Phi_k| vs t
fig = plt.figure(figsize=(9, 6))
for i in range(Nk):
    kx_i, ky_i = modes[i]
    plt.semilogy(t, amp_phi[i, :], label=_mode_label(i, kx_i, ky_i))
plt.xlabel("t")
plt.ylabel(r"$|\phi_k|$")
plt.title(r"Triad amplitudes: $|\phi_k|$ vs time")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(os.path.join(datadir, f"phi_amp_vs_t_{case}.pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# |P_k| vs t
fig = plt.figure(figsize=(9, 6))
for i in range(Nk):
    kx_i, ky_i = modes[i]
    plt.semilogy(t, amp_P[i, :], label=_mode_label(i, kx_i, ky_i))
plt.xlabel("t")
plt.ylabel(r"$|P_k|$")
plt.title(r"Triad amplitudes: $|P_k|$ vs time")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(os.path.join(datadir, f"P_amp_vs_t_{case}.pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# |np.conj(Phi_p)*np.conj(P_q) - np.conj(Phi_q)*np.conj(P_p)| vs t
fig = plt.figure(figsize=(9, 6))
triad_product = np.conj(Phik[2, :]) * np.conj(Pk[0, :]) - np.conj(Phik[0, :]) * np.conj(Pk[2, :])
plt.plot(t, np.abs(triad_product), label="Triad product")
plt.xlabel("t")
plt.ylabel(r"$|\phi_p^* P_q^* - \phi_q^* P_p^*|$")
plt.title(r"Triad interaction: $|\phi_p^* P_q^* - \phi_q^* P_p^*|$ vs time")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(os.path.join(datadir, f"triad_product_vs_t_{case}.pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# phase(Phi_k) vs t
fig = plt.figure(figsize=(9, 6))
for i in range(Nk):
    kx_i, ky_i = modes[i]
    plt.plot(t, phase_phi[i, :], label=_mode_label(i, kx_i, ky_i))
plt.xlabel("t")
plt.ylabel(r"$\arg(\phi_k)$")
plt.title(r"Triad phases: $\arg(\phi_k)$ vs time")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(os.path.join(datadir, f"phi_phase_vs_t_{case}.pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# phase(P_k) vs t
fig = plt.figure(figsize=(9, 6))
for i in range(Nk):
    kx_i, ky_i = modes[i]
    plt.plot(t, phase_P[i, :], label=_mode_label(i, kx_i, ky_i))
plt.xlabel("t")
plt.ylabel(r"$\arg(P_k)$")
plt.title(r"Triad phases: $\arg(P_k)$ vs time")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(os.path.join(datadir, f"P_phase_vs_t_{case}.pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)

# phase difference arg(phi) - arg(P) for q,k,p
fig = plt.figure(figsize=(9, 6))
for i in range(min(3, Nk)):
    kx_i, ky_i = modes[i]
    plt.plot(t, phase_diff[i,:], label=_mode_label(i, kx_i, ky_i))
plt.xlabel("t")
plt.ylabel(r"$\arg(\phi_k) - \arg(P_k)$")
plt.title(r"Phase difference: $\arg(\phi_k) - \arg(P_k)$ (q,k,p)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(os.path.join(datadir, f"phi_minus_P_phase_vs_t_qkp_{case}.pdf"), bbox_inches="tight")
plt.show()
plt.close(fig)