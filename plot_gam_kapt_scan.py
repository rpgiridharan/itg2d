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

kapn=0.2 
kapb=0.02
# Load datasets
fname = datadir+f'gammax_vals_kapt_scan_kapn_{str(kapn).replace(".", "_")}_kapb_{str(kapb).replace(".", "_")}_itg2d.h5'
with h5py.File(fname, 'r') as fl:
    gammax_kapt = fl['gammax_vals'][:]
    kapt_vals = fl['kapt_vals'][:]
    kapb = fl['kapb'][()]

#%% Colormesh of gam(kapt,kz)

plt.figure()
plt.plot(kapt_vals, gammax_kapt)
plt.xlabel('$\\kappa_T$')
plt.ylabel('$\\gamma_{max}$')
plt.title(f"$\\gamma_{{max}}$ for $\\kappa_n$={kapn:.2f} $\\kappa_B$={kapb:.2f}")
plt.savefig(fname.replace('gammax_vals_kapt', 'gammax_kapt').replace('.h5', '.png'), dpi=100)
plt.tight_layout()
plt.show()