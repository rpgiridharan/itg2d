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
    name = ("q", "k", "p")[i] if i < 3 else f"m{i}"
    return f"{name}: (kx,ky)=({kx:.6g},{ky:.6g})"

#%% Load HDF5 file

datadir = "data_instability/"
case_name = "P_pump"
file_name = os.path.join(datadir, f"out_instability_{case_name}.h5")

with h5.File(file_name, "r", swmr=True) as fl:
    t = np.asarray(fl["fields/t"][:], dtype=float)
    Omk = np.asarray(fl["fields/Omk"][:])
    Pk = np.asarray(fl["fields/Pk"][:])
    kx = np.asarray(fl["data/kx"][:], dtype=float)
    ky = np.asarray(fl["data/ky"][:], dtype=float)

Nk = Omk.shape[1]
kpsq = kx**2 + ky**2

Phik = -Omk / kpsq[None, :]

amp_phi = np.abs(Phik)
amp_P = np.abs(Pk)

phase_phi = np.unwrap(np.angle(Phik), axis=0)
phase_P = np.unwrap(np.angle(Pk), axis=0)
# Phase difference: arg(phi * conj(P))
phase_diff = np.unwrap(np.angle(Phik * np.conj(Pk)), axis=0)

#%% Plotting

# |Phi_k| vs t
fig = plt.figure(figsize=(9, 6))
for i in range(Nk):
    plt.semilogy(t, amp_phi[:, i], label=_mode_label(i, kx[i], ky[i]))
plt.xlabel("t")
plt.ylabel(r"$|\phi_k|$")
plt.title(r"Triad amplitudes: $|\phi_k|$ vs time")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(os.path.join(datadir, f"phi_amp_vs_t_{case_name}.png"), bbox_inches="tight")
plt.show()
plt.close(fig)

# |P_k| vs t
fig = plt.figure(figsize=(9, 6))
for i in range(Nk):
    plt.semilogy(t, amp_P[:, i], label=_mode_label(i, kx[i], ky[i]))
plt.xlabel("t")
plt.ylabel(r"$|P_k|$")
plt.title(r"Triad amplitudes: $|P_k|$ vs time")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(os.path.join(datadir, f"P_amp_vs_t_{case_name}.png"), bbox_inches="tight")
plt.show()
plt.close(fig)

# phase(Phi_k) vs t
fig = plt.figure(figsize=(9, 6))
for i in range(Nk):
    plt.plot(t, phase_phi[:, i], label=_mode_label(i, kx[i], ky[i]))
plt.xlabel("t")
plt.ylabel(r"$\arg(\phi_k)$")
plt.title(r"Triad phases: $\arg(\phi_k)$ vs time")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(os.path.join(datadir, f"phi_phase_vs_t_{case_name}.png"), bbox_inches="tight")
plt.show()
plt.close(fig)

# phase(P_k) vs t
fig = plt.figure(figsize=(9, 6))
for i in range(Nk):
    plt.plot(t, phase_P[:, i], label=_mode_label(i, kx[i], ky[i]))
plt.xlabel("t")
plt.ylabel(r"$\arg(P_k)$")
plt.title(r"Triad phases: $\arg(P_k)$ vs time")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(os.path.join(datadir, f"P_phase_vs_t_{case_name}.png"), bbox_inches="tight")
plt.show()
plt.close(fig)

# phase difference arg(phi) - arg(P) for q,k,p
fig = plt.figure(figsize=(9, 6))
for i in range(min(3, Nk)):
    plt.plot(t, phase_diff[:, i], label=_mode_label(i, kx[i], ky[i]))
plt.xlabel("t")
plt.ylabel(r"$\arg(\phi_k) - \arg(P_k)$")
plt.title(r"Phase difference: $\arg(\phi_k) - \arg(P_k)$ (q,k,p)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
os.makedirs(datadir, exist_ok=True)
fig.savefig(os.path.join(datadir, f"phi_minus_P_phase_vs_t_qkp_{case_name}.png"), bbox_inches="tight")
plt.show()
plt.close(fig)