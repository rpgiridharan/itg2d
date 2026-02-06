#%% Import modules
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
file_name = datadir + f'negdiscmax_vals_kapn_kapt_scan_kapb_{str(kapb).replace(".", "_")}_itg2d.h5'
# file_name = datadir + f'negdiscmax_vals_kapn_kapt_scan_kapb_{str(kapb).replace(".", "_")}_itg2d_wo_FLR.h5'

# Load datasets
with h5py.File(file_name, 'r') as fl:
    # Only negdiscmax is required now
    negdiscmax_kapn_kapt = fl['negdiscmax_vals'][:]
    kapn_vals = fl['kapn_vals'][:]
    kapt_vals = fl['kapt_vals'][:]
    kapb = fl['kapb'][()]

# negdiscmax > 0 indicates Δ is negative somewhere on k-grid
positive_any = negdiscmax_kapn_kapt > 0
kapn_pos = kapn_vals[positive_any.any(axis=0)]
kapt_pos = kapt_vals[positive_any.any(axis=1)]
# print(f"kapn values where max(-Δ) > 0: {kapn_pos}")
# print(f"kapt values where max(-Δ) > 0: {kapt_pos}")

#%% Colormesh of discriminant max over (kapn, kapt)

Kapn, Kapt = np.meshgrid(kapn_vals, kapt_vals)
plt.figure(figsize=(16,9))
# max(-Δ): negative (stable everywhere), positive (Δ<0 somewhere)
data = negdiscmax_kapn_kapt.T
vmax = float(np.nanmax(np.abs(data))) if np.isfinite(data).all() else 1.0
vmin = -vmax
pcm = plt.pcolormesh(Kapn, Kapt, data, vmax=vmax, vmin=vmin, cmap='seismic', rasterized=True, shading='auto')
plt.xlabel('$\\kappa_n$')
plt.ylabel('$\\kappa_T$')
plt.title(f"$\\max(-\\Delta)$ for $\\kappa_B$={kapb:.2f}")
plt.axhline(y=0, color='black', linewidth=1, linestyle='-')
plt.axvline(x=0, color='black', linewidth=1, linestyle='-')
plt.colorbar(pcm, label='$\\max(-\\Delta)$')
try:
    CS = plt.contour(Kapn, Kapt, data, levels=[0.0], colors='k', linewidths=2, linestyles='--')
    proxy = Line2D([0], [0], color='k', lw=2, ls='--', label='max(-Δ)=0')
    plt.legend(handles=[proxy], loc='best')
except Exception:
    pass
plt.tight_layout()
plt.savefig(file_name.replace('.h5', '.pdf'))
plt.show()
