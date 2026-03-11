#%% Importing libraries
import numpy as np
import h5py as h5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

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

#%% Load computed spectra
# Npx=512
Npx=1024
datadir=f'data/{Npx}/'

fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

savefile = fname.replace('out_', 'spectrum_')
with h5.File(savefile, 'r') as fl:
    k         = fl['k'][:]
    Phi2k     = fl['Phi2k'][:];     Phi2k_ZF     = fl['Phi2k_ZF'][:];     Phi2k_turb     = fl['Phi2k_turb'][:]
    P2k       = fl['P2k'][:];       P2k_ZF       = fl['P2k_ZF'][:];       P2k_turb       = fl['P2k_turb'][:]
    Ek        = fl['Ek'][:];        Ek_ZF        = fl['Ek_ZF'][:];        Ek_turb        = fl['Ek_turb'][:]
    Kk        = fl['Kk'][:];        Kk_ZF        = fl['Kk_ZF'][:];        Kk_turb        = fl['Kk_turb'][:]
    Wk        = fl['Wk'][:];        Wk_ZF        = fl['Wk_ZF'][:];        Wk_turb        = fl['Wk_turb'][:]
    Gk        = fl['Gk'][:];        Gk_ZF        = fl['Gk_ZF'][:];        Gk_turb        = fl['Gk_turb'][:]
    GKk       = fl['GKk'][:];       GKk_ZF       = fl['GKk_ZF'][:];       GKk_turb       = fl['GKk_turb'][:]

flux_file = fname.replace('out_', 'energy_flux_')
with h5.File(flux_file, 'r') as fl:
    k_f       = float(fl['k_f'][()])
    k_lin     = float(fl['k_lin'][()])

#%% Plots

k1 = np.argmin(np.abs(k - 1))

# plt.figure(figsize=(16, 9))
# plt.loglog(k[1:-1], Phi2k[1:-1], label = r'$\left|\phi_{k}\right|^2$')
# plt.loglog(k[Phi2k_ZF>0][1:-1], Phi2k_ZF[Phi2k_ZF>0][1:-1], label = r'$\left|\phi_{k,\mathrm{ZF}}\right|^2$')
# plt.loglog(k[1:-1], Phi2k_turb[1:-1], label = r'$\left|\phi_{k,\mathrm{turb}}\right|^2$')
# plt.loglog(k[1:-1], Phi2k[k1]*k[1:-1]**(-6), 'r--', label = '$k^{-6}$')
# plt.loglog(k[1:-1], Phi2k[k1]*k[1:-1]**(-8), 'k--', label = '$k^{-8}$')
# plt.xlabel('$k$')
# plt.ylabel(r'$\left|\phi_{k}\right|^2$')
# plt.legend()
# plt.grid(which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'potential_spectrum.pdf', dpi=100)
# else:
#     plt.savefig(datadir+"potential_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
# plt.show()

plt.figure(figsize=(16, 9))
plt.loglog(k[1:-1], P2k[1:-1], label = r'$\left|P_{k}\right|^2$')
plt.loglog(k[P2k_ZF>0][1:-1], P2k_ZF[P2k_ZF>0][1:-1], label = r'$\left|P_{k,\mathrm{ZF}}\right|^2$')
plt.loglog(k[1:-1], P2k_turb[1:-1], label = r'$\left|P_{k,\mathrm{turb}}\right|^2$')
plt.loglog(k[1:-1], P2k[k1]*k[1:-1]**(-1), 'r--', label = '$k^{-1}$')
plt.loglog(k[1:-1], P2k[k1]*k[1:-1]**(-6), 'r--', label = '$k^{-6}$')
plt.loglog(k[1:-1], P2k[k1]*k[1:-1]**(-8), 'k--', label = '$k^{-8}$')
plt.xlabel('$k$')
plt.ylabel(r'$\left|P_k\right|^2$')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'pressure_spectrum.pdf', dpi=100)
else:
    plt.savefig(datadir+"pressure_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

plt.figure(figsize=(16, 9))
plt.loglog(k[1:-1], Ek[1:-1], label = '$E_{k}$')
plt.loglog(k[Ek_ZF>0][1:-1], Ek_ZF[Ek_ZF>0][1:-1], label = '$E_{k,\mathrm{ZF}}$')
plt.loglog(k[1:-1], Ek_turb[1:-1], label = '$E_{k,\mathrm{turb}}$')
plt.loglog(k[1:-1], Ek[k1]*k[1:-1]**(-5/3), 'r--', label = '$k^{-5/3}$')
plt.loglog(k[1:-1], Ek[k1]*k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
plt.loglog(k[1:-1], Ek[k1]*k[1:-1]**(-5), 'k--', label = '$k^{-5}$')

plt.xlabel('$k$')
plt.ylabel('$E_k$')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'energy_spectrum.pdf', dpi=100)
else:
    plt.savefig(datadir+"energy_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# plt.figure(figsize=(16, 9))
# plt.loglog(k[1:-1], Kk[1:-1], label = '$E_{kin,k}$')
# plt.loglog(k[Kk_ZF>0][1:-1], Kk_ZF[Kk_ZF>0][1:-1], label = '$E_{kin,k,\mathrm{ZF}}$')
# plt.loglog(k[1:-1], Kk_turb[1:-1], label = '$E_{kin,k,\mathrm{turb}}$')
# plt.loglog(k[1:-1], Kk[k1]*k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
# plt.loglog(k[1:-1], Kk[k1]*k[1:-1]**(-5), 'k--', label = '$k^{-5}$')
# plt.xlabel('$k$')
# plt.ylabel('$E_{kin,k}$')
# plt.legend()
# plt.grid(which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'kinetic_energy_spectrum.pdf', dpi=100)
# else:
#     plt.savefig(datadir+"kinetic_energy_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
# plt.show()

# plt.figure(figsize=(16, 9))
# plt.loglog(k[1:-1], Wk[1:-1], label = '$\W_{k}$')
# plt.loglog(k[Wk_ZF>0][1:-1], Wk_ZF[Wk_ZF>0][1:-1], label = '$\W_{k,\mathrm{ZF}}$')
# plt.loglog(k[1:-1], Wk_turb[1:-1], label = '$\W_{k,\mathrm{turb}}$')
# plt.loglog(k[1:-1], Wk[k1]*k[1:-1]**(-1), 'r--', label = '$k^{-1}$')
# plt.loglog(k[1:-1], Wk[k1]*k[1:-1]**(-1), 'r--', label = '$k^{-3}$')
# plt.xlabel('$k$')
# plt.ylabel('$\W_k$')
# plt.legend()
# plt.grid(which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'enstrophy_spectrum.pdf', dpi=100)
# else:
#     plt.savefig(datadir+"enstrophy_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
# plt.show()

plt.figure(figsize=(16, 9))
plt.loglog(k[1:-1], Gk[1:-1], label = '$G_{k}$')
plt.loglog(k[Gk_ZF>0][1:-1], Gk_ZF[Gk_ZF>0][1:-1], label = '$G_{k,\mathrm{ZF}}$')
plt.loglog(k[1:-1], Gk_turb[1:-1], label = '$G_{k,\mathrm{turb}}$')
plt.loglog(k[1:-1], Gk[k1]*k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
plt.loglog(k[1:-1], Gk[k1]*k[1:-1]**(-5), 'k--', label = '$k^{-5}$')
plt.xlabel('$k$')
plt.ylabel('$G_{k}$')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'generalized_energy_spectrum.pdf', dpi=100)
else:
    plt.savefig(datadir+"generalized_energy_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# plt.figure(figsize=(16, 9))
# plt.loglog(k[1:-1], GKk[1:-1], label = '$G_{kin,k}$')
# plt.loglog(k[GKk_ZF>0][1:-1], GKk_ZF[GKk_ZF>0][1:-1], label = '$G_{kin,k,\mathrm{ZF}}$')
# plt.loglog(k[1:-1], GKk_turb[1:-1], label = '$G_{kin,k,\mathrm{turb}}$')
# plt.loglog(k[1:-1], GKk[k1]*k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
# plt.loglog(k[1:-1], GKk[k1]*k[1:-1]**(-5), 'k--', label = '$k^{-5}$')
# plt.xlabel('$k$')
# plt.ylabel('$G_{kin,k}$')
# plt.legend()
# plt.grid(which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'generalized_kinetic_energy_spectrum.pdf', dpi=100)
# else:
#     plt.savefig(datadir+"generalized_kinetic_energy_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
# plt.show()

# %%
