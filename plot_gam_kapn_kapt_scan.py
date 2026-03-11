#%% Import modules
import gc
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch 
import h5py

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

#%% Initialize

datadir='data_linear/'
os.makedirs(datadir, exist_ok=True)

kapb=0.02
fname = datadir + f'lin_kapn_kapt_scan_kapb_{str(kapb).replace(".", "_")}_itg2d.h5'
# fname = datadir + f'lin_kapn_kapt_scan_kapb_{str(kapb).replace(".", "_")}_itg2d_wo_FLR.h5'
base_name = fname.replace(datadir+'lin_', '').replace('_scan', '').replace('.h5', '.pdf')

# Load datasets
with h5py.File(fname, 'r') as fl:
    gammax_kapn_kapt = fl['gammax_vals'][:]
    Dturbmax_kapn_kapt = fl['Dturbmax_vals'][:]
    kapn_vals = fl['kapn_vals'][:]
    kapt_vals = fl['kapt_vals'][:]
    kapb = fl['kapb'][()]

Kapn, Kapt = np.meshgrid(kapn_vals, kapt_vals)
zero_or_negative = gammax_kapn_kapt <= 0
kapn_zero = kapn_vals[zero_or_negative.any(axis=0)]
kapt_zero = kapt_vals[zero_or_negative.any(axis=1)]
print(f"kapn values where gamma <= 0: {kapn_zero}")
print(f"kapt values where gamma <= 0: {kapt_zero}")

#%% Colormesh of gam(kapn,kapt)

plt.figure(figsize=(16, 9))
gammax_vmax = np.max(np.abs(gammax_kapn_kapt))
im_gam = plt.pcolormesh(Kapn, Kapt, gammax_kapn_kapt.T, vmax=gammax_vmax, vmin=-gammax_vmax, cmap='seismic', rasterized=True, shading='auto')
plt.contour(Kapn, Kapt, gammax_kapn_kapt.T, levels=[0.0], colors='k', linewidths=2)
plt.plot([], [], color='k', linewidth=2, label=r"$\gamma=0$")
kapn_curve = kapn_vals**2 / (4 * kapb) - kapn_vals
kapn_mask = (kapn_vals < 10 * kapb) & (kapn_curve <= np.max(kapt_vals))
# kapn_mask = (kapn_curve <= np.max(kapt_vals)) #w.o. FLR
plt.plot(kapn_vals[kapn_mask], kapn_curve[kapn_mask], label=r"$\kappa_T=\kappa_n^2/4\kappa_B - \kappa_n$", color='k', linestyle='--', linewidth=2)
plt.axhline(y=0, linewidth=1, color='black')
plt.axvline(x=0, linewidth=1, color='black')
plt.xlabel('$\\kappa_n$')
plt.ylabel('$\\kappa_T$')
plt.legend()
plt.colorbar(im_gam)
plt.savefig(datadir + fname.replace(datadir+'lin_', 'gammax_').replace('.h5', '.pdf'))
plt.show()

#%% Colormesh of Dturb(kapn,kapt)

plt.figure(figsize=(16, 9))
Dturbmax_vmax = np.max(np.abs(Dturbmax_kapn_kapt))
im_dturb = plt.pcolormesh(Kapn, Kapt, Dturbmax_kapn_kapt.T, vmax=Dturbmax_vmax, vmin=-Dturbmax_vmax, cmap='seismic', rasterized=True, shading='auto')
plt.contour(Kapn, Kapt, Dturbmax_kapn_kapt.T, levels=[0.0], colors='k', linewidths=2)
plt.plot([], [], color='k', linewidth=2, label=r"$D_\mathrm{turb}=0$")
plt.axhline(y=0, linewidth=1, color='black')
plt.axvline(x=0, linewidth=1, color='black')
plt.xlabel('$\\kappa_n$')
plt.ylabel('$\\kappa_T$')
plt.legend()
plt.colorbar(im_dturb)
plt.savefig(datadir + fname.replace(datadir+'lin_', 'Dturbmax_').replace('.h5', '.pdf'))
plt.show()