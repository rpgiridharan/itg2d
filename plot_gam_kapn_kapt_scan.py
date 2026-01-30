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
file_name = datadir + f'gammax_vals_kapn_kapt_scan_itg2d_kapb_{str(kapb).replace(".", "_")}.h5'

# Load datasets
with h5py.File(file_name, 'r') as fl:
    gammax_kapn_kapt = fl['gammax_vals'][:]
    kapn_vals = fl['kapn_vals'][:]
    kapt_vals = fl['kapt_vals'][:]
    kapb = fl['kapb'][()]

print(gammax_kapn_kapt == 0)
zero_or_negative = gammax_kapn_kapt <= 0
kapn_zero = kapn_vals[zero_or_negative.any(axis=0)]
kapt_zero = kapt_vals[zero_or_negative.any(axis=1)]
print(f"kapn values where gamma <= 0: {kapn_zero}")
print(f"kapt values where gamma <= 0: {kapt_zero}")
#%% Colormesh of gam(kapt,kz)

Kapn, Kapt = np.meshgrid(kapn_vals, kapt_vals)
plt.figure()
plt.pcolormesh(Kapn, Kapt, gammax_kapn_kapt.T, vmax=1.0, vmin=-1.0, cmap='seismic', rasterized=True, shading='auto')
plt.xlabel('$\\kappa_n$')
plt.ylabel('$\\kappa_T$')
plt.title(f"$\\gamma_{{max}}$ for $\\kappa_B$={kapb:.2f}")
plt.colorbar()
plt.savefig(datadir + 'gammax_kapn_kapt_itg2d.pdf')
plt.show()