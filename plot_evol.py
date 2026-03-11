#%% Importing libraries
import h5py as h5
import numpy as np
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

#%% Load the computed HDF5 file (produced by compute_evol.py)

# Npx=512
Npx=1024
datadir=f'data/{Npx}/'

# fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

evol_fname = datadir + fname.split('/')[-1].replace('out_', 'evol_')

with h5.File(evol_fname, 'r') as fl:
    t                        = fl['t'][:]
    P2_t                     = fl['P2_t'][:]
    P2_ZF_t                  = fl['P2_ZF_t'][:]
    energy_t                 = fl['energy_t'][:]
    energy_ZF_t              = fl['energy_ZF_t'][:]
    kin_energy_t             = fl['kin_energy_t'][:]
    kin_energy_ZF_t          = fl['kin_energy_ZF_t'][:]
    enstrophy_t              = fl['enstrophy_t'][:]
    enstrophy_ZF_t           = fl['enstrophy_ZF_t'][:]
    gen_energy_t             = fl['gen_energy_t'][:]
    gen_energy_ZF_t          = fl['gen_energy_ZF_t'][:]
    entropy_t                = fl['entropy_t'][:]
    Ombar_t                  = fl['Ombar_t'][:]
    Qbox_t                   = fl['Qbox_t'][:]
    electric_reynolds_power_t   = fl['electric_reynolds_power_t'][:]
    diamagnetic_reynolds_power_t = fl['diamagnetic_reynolds_power_t'][:]
    reynolds_power_t         = fl['reynolds_power_t'][:]

nt = len(t)

#%% Calculate derived turbulent quantities and plot

P2_turb_t = P2_t - P2_ZF_t
energy_turb_t = energy_t - energy_ZF_t
kin_energy_turb_t = kin_energy_t - kin_energy_ZF_t
enstrophy_turb_t = enstrophy_t - enstrophy_ZF_t
gen_energy_turb_t = gen_energy_t - gen_energy_ZF_t

# Plot variance(P) vs time
plt.figure(figsize=(16, 9))
plt.semilogy(t, P2_t, label = r'$P_{\mathrm{total}}^2$')
plt.semilogy(t, P2_ZF_t, label = r'$P_{\mathrm{ZF}}^2$')
plt.semilogy(t, P2_turb_t, label = r'$P_{\mathrm{turb}}^2$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$\langle P^2\rangle$')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'P2_vs_t.pdf',dpi=100)
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'P2_vs_t_').replace('.h5', '.pdf'),dpi=100)
plt.show()

# Plot total energy vs time
plt.figure(figsize=(16, 9))
plt.semilogy(t, energy_t, label = r'$E_{\mathrm{total}}$')
plt.semilogy(t, energy_ZF_t, label = r'$E_{\mathrm{ZF}}$')
plt.semilogy(t, energy_turb_t, label = r'$E_{\mathrm{turb}}$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$E$')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'energy_vs_t.pdf',dpi=100)
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'energy_vs_t_').replace('.h5', '.pdf'),dpi=100)
plt.show()

# Plot zonal energy fraction vs time
plt.figure(figsize=(16, 9))
plt.semilogy(t, energy_ZF_t/energy_t, label = r'$E_{\mathrm{ZF}}/E$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$E_{\mathrm{ZF}}/E$')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'zonal_energy_fraction_vs_t.pdf',dpi=100)
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'zonal_energy_fraction_vs_t_').replace('.h5', '.pdf'),dpi=100)
plt.show()

# # Plot kinetic energy vs time
# plt.figure(figsize=(8,6))
# plt.semilogy(t, kin_energy_t, label = r'$E_{\mathrm{kin,\mathrm{total}}}$')
# plt.semilogy(t, kin_energy_ZF_t, label = r'$E_{\mathrm{kin,\mathrm{ZF}}}$')
# plt.semilogy(t, kin_energy_turb_t, label = r'$E_{\mathrm{kin,\mathrm{turb}}}$')
# plt.xlabel(r'$\gamma t$')
# plt.ylabel(r'$E_{\mathrm{kin}}$')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'kinetic_energy_vs_t.pdf',dpi=100)
# else:
#     plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'kinetic_energy_vs_t_').replace('.h5', '.pdf'),dpi=100)
# plt.show()

# Plot generalized energy vs time
plt.figure(figsize=(16, 9))
plt.semilogy(t, gen_energy_t, label = r'$G$')
plt.semilogy(t, gen_energy_ZF_t, label = r'$G_{\mathrm{ZF}}$')
plt.semilogy(t, gen_energy_turb_t, label = r'$G_{\mathrm{turb}}$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$G$')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'generalized_energy_vs_t.pdf',dpi=100)
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'generalized_energy_vs_t_').replace('.h5', '.pdf'),dpi=100)
plt.show()

# # Plot hyd. entropy vs time
# plt.figure(figsize=(8,6))
# plt.semilogy(t, entropy_t, label = r'$S$')
# plt.xlabel(r'$\gamma t$')
# plt.ylabel(r'$S=-\sum_{\mathbf{k}}p_{\mathbf{k}}\log p_{\mathbf{k}}$')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'entropy_vs_t.pdf',dpi=100)
# else:
#     plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'entropy_vs_t_').replace('.h5', '.pdf'), dpi=100)
# plt.show()

# Plot Q vs time
plt.figure(figsize=(16, 9))
plt.plot(t, Qbox_t, '-', label = r'$Q_{\mathrm{box}}$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$Q_{\mathrm{box}}$')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'Qbox_vs_t.pdf',dpi=100)
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'Qbox_vs_t_').replace('.h5', '.pdf'), dpi=100)
plt.show()

# Plot Reynolds power vs time
plt.figure(figsize=(16, 9))
plt.plot(t, electric_reynolds_power_t, '-', label = r'$<R_{\mathrm{\phi}} \partial_x \bar{v}_y>$')
plt.plot(t, diamagnetic_reynolds_power_t, '-', label = r'$<R_{\mathrm{d}}  \partial_x \bar{v}_y>$')
plt.plot(t, reynolds_power_t, '-', label = r'$<R \partial_x \bar{v}_y>$')
plt.xlabel(r'$\gamma t$')
plt.ylabel('Reynolds power')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'reynolds_power_vs_t.pdf',dpi=100)
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'reynolds_power_vs_t_').replace('.h5', '.pdf'), dpi=100)
plt.show()

# Plot Cumulative Reynolds power vs time
plt.figure(figsize=(16, 9))
plt.plot(t, np.cumsum(electric_reynolds_power_t), '-', label = r'$<R_{\mathrm{\phi}} \partial_x \bar{v}_y>$')
plt.plot(t, np.cumsum(diamagnetic_reynolds_power_t), '-', label = r'$<R_{\mathrm{d}}  \partial_x \bar{v}_y>$')
plt.plot(t, np.cumsum(reynolds_power_t), '-', label = r'$<R \partial_x \bar{v}_y>$')
plt.xlabel(r'$\gamma t$')
plt.ylabel('Cumulative Reynolds power')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'cum_reynolds_power_vs_t.pdf',dpi=100)
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'cum_reynolds_power_vs_t_').replace('.h5', '.pdf'), dpi=100)
plt.show()