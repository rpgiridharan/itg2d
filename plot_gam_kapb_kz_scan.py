#%% Import modules
import gc
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch 
import h5py

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 3  
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.width'] = 1.5 
plt.rcParams['ytick.minor.width'] = 1.5 

#%% Initialize

datadir='data_linear/'
os.makedirs(datadir, exist_ok=True)

# Load datasets
with h5py.File(datadir + 'gammax_vals_kapb_kz_scan_itg2d3c.h5', 'r') as fl:
    gammax_kapb_kz = fl['gammax_vals'][:]
    kapb_vals = fl['kapb_vals'][:]
    kz_vals = fl['kz_vals'][:]
    kapt = fl['kapt'][()]

#%% Colormesh of gam(kapt,kz)

Kapb, Kz = np.meshgrid(kapb_vals, kz_vals)
plt.figure()
plt.pcolormesh(Kapb, Kz, gammax_kapb_kz.T, vmax=1.0, vmin=-1.0, cmap='seismic', rasterized=True, shading='auto')
plt.xlabel('$\\kappa_B$')
plt.ylabel('$k_z$')
plt.title(f"$\\gamma_{{max}}$ for $\\kappa_T$={kapt:.2f}")
plt.colorbar()
plt.savefig(datadir + 'gammax_kapb_kz_itg2d3c.png', dpi=600)
plt.show()
del gammax_kapb_kz