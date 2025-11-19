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
with h5py.File(datadir + 'gammax_vals_kapt_scan_itg2d_D.h5', 'r') as fl:
    gammax_kapt = fl['gammax_vals'][:]
    kapt_vals = fl['kapt_vals'][:]
    kapb = fl['kapb'][()]

#%% Colormesh of gam(kapt,kz)

plt.figure()
plt.plot(kapt_vals, gammax_kapt)
plt.xlabel('$\\kappa_T$')
plt.ylabel('$\\gamma_{max}$')
plt.title(f"$\\gamma_{{max}}$ for $\\kappa_B$={kapb:.2f}")
plt.savefig(datadir + 'gammax_kapt_itg2d_D.png', dpi=600)
plt.tight_layout()
plt.show()