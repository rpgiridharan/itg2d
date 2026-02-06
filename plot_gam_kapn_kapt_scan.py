#%% Import modules
import gc
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch 
import h5py

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['axes.linewidth'] = 3  
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.width'] = 1.5 
plt.rcParams['ytick.minor.width'] = 1.5 
plt.rcParams['savefig.dpi'] = 100
plt.rcParams.update({
    "font.size": 22,          # default text
    "axes.titlesize": 30,     # figure title
    "axes.labelsize": 26,     # x/y labels
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22
})

#%% Initialize

datadir='data_linear/'
os.makedirs(datadir, exist_ok=True)

kapb=0.02
# file_name = datadir + f'lin_kapn_kapt_scan_kapb_{str(kapb).replace(".", "_")}_itg2d.h5'
file_name = datadir + f'lin_kapn_kapt_scan_kapb_{str(kapb).replace(".", "_")}_itg2d_wo_FLR.h5'
base_name = file_name.replace(datadir+'lin_', '').replace('_scan', '').replace('.h5', '.pdf')

# Load datasets
with h5py.File(file_name, 'r') as fl:
    gammax_kapn_kapt = fl['gammax_vals'][:]
    Dturbmax_kapn_kapt = fl['Dturbmax_vals'][:]
    kapn_vals = fl['kapn_vals'][:]
    kapt_vals = fl['kapt_vals'][:]
    kapb = fl['kapb'][()]

zero_or_negative = gammax_kapn_kapt <= 0
kapn_zero = kapn_vals[zero_or_negative.any(axis=0)]
kapt_zero = kapt_vals[zero_or_negative.any(axis=1)]
print(f"kapn values where gamma <= 0: {kapn_zero}")
print(f"kapt values where gamma <= 0: {kapt_zero}")
#%% Colormesh of gam(kapt,kz)

Kapn, Kapt = np.meshgrid(kapn_vals, kapt_vals)
plt.figure(figsize=(16,9))
gammax_vmax = np.max(np.abs(gammax_kapn_kapt))
im_gam = plt.pcolormesh(Kapn, Kapt, gammax_kapn_kapt.T, vmax=gammax_vmax, vmin=-gammax_vmax, cmap='seismic', rasterized=True, shading='auto')
plt.contour(Kapn, Kapt, gammax_kapn_kapt.T, levels=[0.0], colors='k', linewidths=2)
plt.axhline(y=0, linewidth=1, color='black')
plt.axvline(x=0, linewidth=1, color='black')
plt.xlabel('$\\kappa_n$')
plt.ylabel('$\\kappa_T$')
plt.title(f"$\\gamma_{{max}}$ for $\\kappa_B$={kapb:.2f}")
plt.colorbar(im_gam)
plt.savefig(datadir + file_name.replace(datadir+'lin_', 'gammax_').replace('.h5', '.pdf'))
plt.show()

#%% Colormesh of Dturb(kapt,kz)

Kapn, Kapt = np.meshgrid(kapn_vals, kapt_vals)
plt.figure(figsize=(16,9))
Dturbmax_vmax = np.max(np.abs(Dturbmax_kapn_kapt))
im_dturb = plt.pcolormesh(Kapn, Kapt, Dturbmax_kapn_kapt.T, vmax=Dturbmax_vmax, vmin=-Dturbmax_vmax, cmap='seismic', rasterized=True, shading='auto')
plt.contour(Kapn, Kapt, Dturbmax_kapn_kapt.T, levels=[0.0], colors='k', linewidths=2)
plt.axhline(y=0, linewidth=1, color='black')
plt.axvline(x=0, linewidth=1, color='black')
plt.xlabel('$\\kappa_n$')
plt.ylabel('$\\kappa_T$')
plt.title(f"$\\left(\\frac{{\\gamma}}{{k^2}}\\right)_{{\\text{{max}}}}$ for $\\kappa_B$={kapb:.2f}")
plt.colorbar(im_dturb)
plt.savefig(datadir + file_name.replace(datadir+'lin_', 'Dturbmax_').replace('.h5', '.pdf'))
plt.show()