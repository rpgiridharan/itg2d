#%% Importing libraries
import numpy as np
import h5py as h5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, skew, kurtosis

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

#%% Load computed flux data

# Npx=512
Npx=1024
datadir=f'data/{Npx}/'

# fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

flux_file = fname.replace('out_', 'energy_flux_')
with h5.File(flux_file, 'r') as fl:
    k         = fl['k'][:]
    k_f       = float(fl['k_f'][()])
    k_lin     = float(fl['k_lin'][()])
    Pik       = fl['Pik'][:]
    Pik_phi   = fl['Pik_phi'][:]
    Pik_d     = fl['Pik_d'][:]
    fk        = fl['fk'][:]
    dk        = fl['dk'][:]
    Pik_phi_t = fl['Pik_phi_t'][:]
    Pik_d_t   = fl['Pik_d_t'][:]

#%% Derived quantities for PDFs

# PDF of fluxes at k_f
idx_k_f = np.argmax(fk)
Pik_phi_series = Pik_phi_t[:, idx_k_f]
Pik_d_series   = Pik_d_t[:, idx_k_f]
Pik_series     = Pik_phi_series + Pik_d_series

Pik_phi_series_norm = (Pik_phi_series - np.mean(Pik_phi_series)) / np.std(Pik_phi_series)
Pik_d_series_norm   = (Pik_d_series - np.mean(Pik_d_series)) / np.std(Pik_d_series)
Pik_series_norm     = (Pik_series - np.mean(Pik_series)) / np.std(Pik_series)

# PDF of fluxes at k=kymax
idx_k_lin = np.argmin(np.abs(k - k_lin))
Pik_phi_series_max = Pik_phi_t[:, idx_k_lin]
Pik_d_series_max   = Pik_d_t[:, idx_k_lin]
Pik_series_max     = Pik_phi_series_max + Pik_d_series_max

Pik_phi_series_max_norm = (Pik_phi_series_max - np.mean(Pik_phi_series_max)) / np.std(Pik_phi_series_max)
Pik_d_series_max_norm   = (Pik_d_series_max - np.mean(Pik_d_series_max)) / np.std(Pik_d_series_max)
Pik_series_max_norm     = (Pik_series_max - np.mean(Pik_series_max)) / np.std(Pik_series_max)

# PDF of fluxes at k=1
idx_k_1 = np.argmin(np.abs(k - 1))
Pik_phi_series_1 = Pik_phi_t[:, idx_k_1]
Pik_d_series_1   = Pik_d_t[:, idx_k_1]
Pik_series_1     = Pik_phi_series_1 + Pik_d_series_1

Pik_phi_series_1_norm = (Pik_phi_series_1 - np.mean(Pik_phi_series_1)) / np.std(Pik_phi_series_1)
Pik_d_series_1_norm   = (Pik_d_series_1 - np.mean(Pik_d_series_1)) / np.std(Pik_d_series_1)
Pik_series_1_norm     = (Pik_series_1 - np.mean(Pik_series_1)) / np.std(Pik_series_1)

#%% Plots

plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], Pik[1:-1], label = r'$\Pi_{k}$')
plt.plot(k[1:-1], Pik_phi[1:-1], label = r'$\Pi_{k,\mathrm{\phi}}$')
plt.plot(k[1:-1], Pik_d[1:-1], label = r'$\Pi_{k,\mathrm{d}}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2, label=f'$k_{{lin}}={k_lin:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\Pi_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'energy_flux.pdf', dpi=100)
else:
    plt.savefig(datadir+"energy_flux_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], fk[1:-1], label = r'$f_{k,\mathrm{total}}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$f_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'energy_injection.pdf', dpi=100)
else:
    plt.savefig(datadir+"energy_injection_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], dk[1:-1], label = r'$d_{k,\mathrm{total}}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$d_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'dissipation.pdf', dpi=100)
else:
    plt.savefig(datadir+"dissipation_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# PDF of fluxes at k_f
plt.figure(figsize=(16, 9))
for series, label, color in zip([Pik_series_norm, Pik_phi_series_norm, Pik_d_series_norm],
                        [r'$\Pi_{k}$', r'$\Pi_{k,\mathrm{\phi}}$', r'$\Pi_{k,\mathrm{d}}$'],
                        ['C0', 'C1', 'C2']):
    s = skew(series)
    f = kurtosis(series, fisher=False)  # Gaussian = 3
    kde = gaussian_kde(series)
    x_range = np.linspace(series.min(), series.max(), 200)
    plt.hist(series, bins=50, density=True, alpha=0.3, color=color)
    plt.plot(x_range, kde(x_range), label=rf'{label}  $S={s:.2f},\ F={f:.2f}$', color=color)
plt.xlabel(r'$\frac{\Pi_k-<\Pi_k>}{\sigma}$')
plt.ylabel('PDF')
plt.gca().text(0.97, 0.97, rf'$k_f={k_f:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'energy_flux_pdf_.pdf', dpi=100)
else:
    plt.savefig(datadir+"energy_flux_pdf_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# PDF of fluxes at k=kymax
plt.figure(figsize=(16, 9))
for series, label, color in zip([Pik_series_max_norm, Pik_phi_series_max_norm, Pik_d_series_max_norm],
                        [r'$\Pi_{k}$', r'$\Pi_{k,\mathrm{\phi}}$', r'$\Pi_{k,\mathrm{d}}$'],
                        ['C0', 'C1', 'C2']):
    s = skew(series)
    f = kurtosis(series, fisher=False)  # Gaussian = 3
    kde = gaussian_kde(series)
    x_range = np.linspace(series.min(), series.max(), 200)
    plt.hist(series, bins=50, density=True, alpha=0.3, color=color)
    plt.plot(x_range, kde(x_range), label=rf'{label}  $S={s:.2f},\ F={f:.2f}$', color=color)
plt.xlabel(r'$\frac{\Pi_k-<\Pi_k>}{\sigma}$')
plt.ylabel('PDF')
plt.gca().text(0.97, 0.97, rf'$k_{{lin}}={k_lin:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'energy_flux_kymax_pdf_.pdf', dpi=100)
else:
    plt.savefig(datadir+"energy_flux_kymax_pdf_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# PDF of fluxes at k=1
plt.figure(figsize=(16, 9))
for series, label, color in zip([Pik_series_1_norm, Pik_phi_series_1_norm, Pik_d_series_1_norm],
                        [r'$\Pi_{k}$', r'$\Pi_{k,\mathrm{\phi}}$', r'$\Pi_{k,\mathrm{d}}$'],
                        ['C0', 'C1', 'C2']):
    s = skew(series)
    f = kurtosis(series, fisher=False)  # Gaussian = 3
    kde = gaussian_kde(series)
    x_range = np.linspace(series.min(), series.max(), 200)
    plt.hist(series, bins=50, density=True, alpha=0.3, color=color)
    plt.plot(x_range, kde(x_range), label=rf'{label}  $S={s:.2f},\ F={f:.2f}$', color=color)
plt.xlabel(r'$\frac{\Pi_k-<\Pi_k>}{\sigma}$')
plt.ylabel('PDF')
plt.gca().text(0.97, 0.97, r'$k=1$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'energy_flux_k1_pdf_.pdf', dpi=100)
else:
    plt.savefig(datadir+"energy_flux_k1_pdf_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()
