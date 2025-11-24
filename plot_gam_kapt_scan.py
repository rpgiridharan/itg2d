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

kapn=0.2 
kapb=0.02
# Load datasets
file_name = datadir+f'gammax_vals_kapt_scan_kapn_{str(kapn).replace(".", "_")}_kapb_{str(kapb).replace(".", "_")}_itg2d.h5'
with h5py.File(file_name, 'r') as fl:
    gammax_kapt = fl['gammax_vals'][:]
    kapt_vals = fl['kapt_vals'][:]
    kapb = fl['kapb'][()]

#%% Colormesh of gam(kapt,kz)

plt.figure()
plt.plot(kapt_vals, gammax_kapt)
plt.xlabel('$\\kappa_T$')
plt.ylabel('$\\gamma_{max}$')
plt.title(f"$\\gamma_{{max}}$ for $\\kappa_n$={kapn:.2f} $\\kappa_B$={kapb:.2f}")
plt.savefig(file_name.replace('gammax_vals_kapt', 'gammax_kapt').replace('.h5', '.png'), dpi=600)
plt.tight_layout()
plt.show()