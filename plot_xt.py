#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np,irftnp,rftnp
from modules.plot_basics import symmetrize_y_axis
from modules.gamma import gam_max
import os
import glob 

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

#%% Load the HDF5 file

Npx=512
datadir=f'data/{Npx}/'

# fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

# kapt=0.2
# D=0.1
# pattern = datadir + f'out_kapt_{str(kapt).replace(".", "_")}_D_{str(D).replace(".", "_")}*_1024_1024.h5'
# files = glob.glob(pattern)
# if not files:
#     print(f"No file found for kappa_T = {kapt}")
# else:
#     fname = files[0]

# Downsample time axis to reduce memory usage
stride = 4

with h5.File(fname, 'r', swmr=True) as fl:
    RPhi_t = fl['fluxes/RPhi'][::stride].astype(np.float32)
    RP_t = fl['fluxes/RP'][::stride].astype(np.float32)
    vbar_t = fl['zonal/vbar'][::stride].astype(np.float32)
    dxvbar_t = fl['zonal/Ombar'][::stride].astype(np.float32)
    t = fl['fluxes/t'][::stride].astype(np.float32)
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]
    kapn = fl['params/kapn'][()]
    kapt = fl['params/kapt'][()]
    kapb = fl['params/kapb'][()]
    D = fl['params/D'][()]
    if 'H' in fl['params']:
        H = fl['params/H'][()]
    elif 'HP' in fl['params']:
        HP = fl['params/HP'][()]
        H=HP

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
gammax=gam_max(kx,ky,kapn,kapt,kapb,D,H)
t=t*gammax
nt = len(t)

sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]

xl,yl=np.arange(0,Lx,Lx/Npx),np.arange(0,Ly,Ly/Npy)
x,y=np.meshgrid(np.array(xl),np.array(yl),indexing='ij')
nt_data = min(nt, vbar_t.shape[0])
xm, tm = np.meshgrid(x[:, 0], t[:nt_data])

R_t = RPhi_t + RP_t
del RPhi_t, RP_t

vbar_lim   = float(np.percentile(np.abs(vbar_t),75))
dxvbar_lim = float(np.percentile(np.abs(dxvbar_t),75))
R_lim      = float(np.percentile(np.abs(R_t),75))
P_lim      = float(np.percentile(np.abs(R_t[:nt_data] * dxvbar_t[:nt_data]),75))

#%% Overview 2x2

P_t = R_t[:nt_data] * dxvbar_t[:nt_data]
fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
axs[0, 0].pcolormesh(xm, tm, vbar_t[:nt_data,:], vmin=-vbar_lim, vmax=vbar_lim, cmap='seismic')
axs[0, 0].set_title('Zonal flow: $\\partial_x\\overline{\\phi}$')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('$\\gamma t$')

axs[0, 1].pcolormesh(xm, tm, R_t[:nt_data,:], vmin=-R_lim, vmax=R_lim, cmap='seismic')
axs[0, 1].set_title('Reynolds Stress: $R$')
axs[0, 1].set_xlabel('x')

axs[1, 0].pcolormesh(xm, tm, dxvbar_t[:nt_data,:], vmin=-dxvbar_lim, vmax=dxvbar_lim, cmap='seismic')
axs[1, 0].set_title('Zonal shear: $\\partial_x^2\\overline{\\phi}$')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('$\\gamma t$')

axs[1, 1].pcolormesh(xm, tm, P_t[:nt_data,:], vmin=-P_lim, vmax=P_lim, cmap='seismic')
axs[1, 1].set_title('$R$ production: $P$')
axs[1, 1].set_xlabel('x')
del P_t

fig.suptitle(f'$\\kappa_T={kapt}$, $D={D}$', fontsize=20, y=1.03)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'R_and_P_xt_plots.pdf', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'R_and_P_xt_plots_').replace('.h5', '.pdf'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()

#%% Zonal flow

plt.figure(figsize=(16, 9))
plt.pcolormesh(xm, tm, vbar_t[:nt_data,:], vmin=-vbar_lim, vmax=vbar_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('Zonal flow: $\\partial_x\\overline{\\phi}$')
plt.colorbar()
plt.tight_layout(pad=0.5)
if fname.endswith('out.h5'):
    plt.savefig(datadir+'vbar_xt.pdf', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'vbar_xt_').replace('.h5', '.pdf'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()
del vbar_t

#%% Reynolds stresses
# Reload RPhi_t and RP_t from disk (freed earlier to save memory)
with h5.File(fname, 'r', swmr=True) as fl:
    RPhi_t = fl['fluxes/RPhi'][::stride].astype(np.float32)
    RP_t   = fl['fluxes/RP'][::stride].astype(np.float32)

RPhi_lim = float(np.percentile(np.abs(RPhi_t),75))
RP_lim   = float(np.percentile(np.abs(RP_t),75))

fig, axs = plt.subplots(1, 3, figsize=(16, 9), sharey=True)

axs[0].pcolormesh(xm, tm, RPhi_t[:nt_data,:], vmin=-RPhi_lim, vmax=RPhi_lim, cmap='seismic')
axs[0].set_title('$R_\\phi$')
axs[0].set_xlabel('x')
axs[0].set_ylabel('$\\gamma t$')
fig.colorbar(axs[0].collections[0], ax=axs[0])
PPhi_t = RPhi_t[:nt_data] * dxvbar_t[:nt_data]
PPhi_lim = float(np.percentile(np.abs(PPhi_t),75))
del RPhi_t

axs[1].pcolormesh(xm, tm, RP_t[:nt_data,:], vmin=-RP_lim, vmax=RP_lim, cmap='seismic')
axs[1].set_title('$R_d$')
axs[1].set_xlabel('x')
fig.colorbar(axs[1].collections[0], ax=axs[1])
PP_t = RP_t[:nt_data] * dxvbar_t[:nt_data]
PP_lim = float(np.percentile(np.abs(PP_t),75))
del RP_t, dxvbar_t

axs[2].pcolormesh(xm, tm, R_t[:nt_data,:], vmin=-R_lim, vmax=R_lim, cmap='seismic')
axs[2].set_title('$R=R_\\phi + R_d$')
axs[2].set_xlabel('x')
fig.colorbar(axs[2].collections[0], ax=axs[2])
del R_t

fig.suptitle('Reynolds Stress', fontsize=20)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'R_xt.pdf', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'R_xt_').replace('.h5', '.pdf'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()

#%% Reynolds stress production

fig, axs = plt.subplots(1, 3, figsize=(16, 9), sharey=True)

axs[0].pcolormesh(xm, tm, PPhi_t[:nt_data,:], vmin=-PPhi_lim, vmax=PPhi_lim, cmap='seismic')
axs[0].set_title('$P_\\phi$')
axs[0].set_xlabel('x')
axs[0].set_ylabel('$\\gamma t$')
fig.colorbar(axs[0].collections[0], ax=axs[0])

axs[1].pcolormesh(xm, tm, PP_t[:nt_data,:], vmin=-PP_lim, vmax=PP_lim, cmap='seismic')
axs[1].set_title('$P_d$')
axs[1].set_xlabel('x')
fig.colorbar(axs[1].collections[0], ax=axs[1])

P_t = PPhi_t + PP_t
del PPhi_t, PP_t
axs[2].pcolormesh(xm, tm, P_t[:nt_data,:], vmin=-P_lim, vmax=P_lim, cmap='seismic')
axs[2].set_title('$P=P_\\phi + P_d$')
axs[2].set_xlabel('x')
fig.colorbar(axs[2].collections[0], ax=axs[2])
del P_t

fig.suptitle('Reynolds stress production', fontsize=20)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'P_xt.pdf', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'P_xt_').replace('.h5', '.pdf'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()

#%% Heat flux

with h5.File(fname, 'r', swmr=True) as fl:
    Q_t = fl['fluxes/Q'][::stride].astype(np.float32)

Q_lim = float(np.percentile(np.abs(Q_t),90))
plt.figure(figsize=(16, 9))
plt.pcolormesh(xm, tm, Q_t[:nt_data,:], vmin=-Q_lim, vmax=Q_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('Heat flux: $Q$')
plt.colorbar()
plt.tight_layout(pad=0.5)
if fname.endswith('out.h5'):
    plt.savefig(datadir+'Q_xt.pdf', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'Q_xt_').replace('.h5', '.pdf'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()
del Q_t, xm, tm
