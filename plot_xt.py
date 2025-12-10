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

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 3  

#%% Load the HDF5 file

datadir = 'data_scan/'
kapt=0.2
D=0.1
pattern = datadir + f'out_kapt_{str(kapt).replace(".", "_")}_D_{str(D).replace(".", "_")}*.h5'
files = glob.glob(pattern)
if not files:
    print(f"No file found for kappa_T = {kapt}")
else:
    file_name = files[0]

with h5.File(file_name, 'r', swmr=True) as fl:
    Q_t = fl['fluxes/Q'][:]
    RPhi_t = fl['fluxes/RPhi'][:]
    RP_t = fl['fluxes/RP'][:]
    vbar_t = fl['zonal/vbar'][:]
    Ombar_t = fl['zonal/Ombar'][:]
    t = fl['fluxes/t'][:]
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
    HP = fl['params/HP'][()]
    HPhi = fl['params/HPhi'][()]

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
gammax=gam_max(kx,ky,kapn,kapt,kapb,D,HP,HPhi)
t=t*gammax
nt = len(t)

sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]

xl,yl=np.arange(0,Lx,Lx/Npx),np.arange(0,Ly,Ly/Npy)
x,y=np.meshgrid(np.array(xl),np.array(yl),indexing='ij')

dxvbar_t = Ombar_t
R_t = RPhi_t + RP_t
RP_frac_t = RP_t / R_t
PPhi_t = RPhi_t*dxvbar_t
PP_t = RP_t*dxvbar_t
P_t = PPhi_t + PP_t
PP_frac_t = PP_t / P_t

vbar_lim = np.mean(np.abs(vbar_t))
dxvbar_lim = np.mean(np.abs(dxvbar_t))
RPhi_lim = np.mean(np.abs(RPhi_t))
RP_lim = np.mean(np.abs(RP_t))
R_lim = np.mean(np.abs(R_t))
RP_frac_lim = np.mean(np.abs(RP_frac_t))
PPhi_lim = np.mean(np.abs(PPhi_t))
PP_lim = np.mean(np.abs(PP_t))
P_lim = np.mean(np.abs(P_t))
PP_frac_lim = np.mean(np.abs(PP_frac_t))

#%% Plots

# Create meshgrid with proper dimensions for pcolor
nt_data = min(nt, vbar_t.shape[0])  # to not exceed array bounds
xm, tm = np.meshgrid(x[:, 0], t[:nt_data])

fig, axs = plt.subplots(2, 4, figsize=(18, 10),sharey=True)  # 2 rows, 4 columns
axs[0, 0].pcolormesh(xm, tm, vbar_t[:nt_data,:], vmin=-vbar_lim, vmax=vbar_lim, cmap='seismic')
axs[0, 0].set_title('Zonal flow: $\\partial_x\\overline{\\phi}$')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('$\\gamma t$')

axs[0, 1].pcolormesh(xm, tm, RPhi_t[:nt_data,:], vmin=-RPhi_lim, vmax=RPhi_lim, cmap='seismic')
axs[0, 1].set_title('Electric Reynolds Stress: $R_\\phi$')
axs[0, 0].set_xlabel('x')

axs[0, 2].pcolormesh(xm, tm, PP_t[:nt_data,:], vmin=-PP_lim, vmax=PP_lim, cmap='seismic')
axs[0, 2].set_title('Diamagnetic Reynolds Stress: $R_d$')
axs[0, 0].set_xlabel('x')

axs[0, 3].pcolormesh(xm, tm, R_t[:nt_data,:], vmin=-R_lim, vmax=R_lim, cmap='seismic')
axs[0, 3].set_title('Reynolds Stress: $R$')
axs[0, 0].set_xlabel('x')

axs[1, 0].pcolormesh(xm, tm, dxvbar_t[:nt_data,:], vmin=-dxvbar_lim, vmax=dxvbar_lim, cmap='seismic')
axs[1, 0].set_title('Zonal shear: $\\partial_x^2\\overline{\\phi}$')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('$\\gamma t$')

axs[1, 1].pcolormesh(xm, tm, PPhi_t[:nt_data,:], vmin=-PPhi_lim, vmax=PPhi_lim, cmap='seismic')
axs[1, 1].set_title('$R_\\phi$ production: $P_\\phi$')
axs[0, 0].set_xlabel('x')

axs[1, 2].pcolormesh(xm, tm, PP_t[:nt_data,:], vmin=-PP_lim, vmax=PP_lim, cmap='seismic')
axs[1, 2].set_title('$R_d$ production: $P_d$')
axs[0, 0].set_xlabel('x')

axs[1, 3].pcolormesh(xm, tm, P_t[:nt_data,:], vmin=-P_lim, vmax=P_lim, cmap='seismic')
axs[1, 3].set_title('$R$ production: $P$')
axs[0, 0].set_xlabel('x')

fig.suptitle(f'$\\kappa_T={kapt}$, $D={D}$', fontsize=20, y=1.03)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'R_and_P_xt_plots.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'R_and_P_xt_plots_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 5))  
plt.pcolormesh(xm, tm, vbar_t[:nt_data,:], vmin=-vbar_lim, vmax=vbar_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('Zonal flow: $\\partial_x\\overline{\\phi}$')
plt.colorbar()
plt.tight_layout(pad=0.5) 
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'vbar_xt.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'vbar_xt_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 5)) 
plt.pcolormesh(xm, tm, dxvbar_t[:nt_data,:], vmin=-dxvbar_lim, vmax=dxvbar_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('Zonal shear: $\\partial_x^2\\overline{\\phi}$')
plt.colorbar()
plt.tight_layout(pad=0.5) 
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'Ombar_xt.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'Ombar_xt_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 5))  
plt.pcolormesh(xm, tm, RPhi_t[:nt_data,:], vmin=-RPhi_lim, vmax=RPhi_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('Electric Reynolds Stress: $R_\\phi$')
plt.colorbar()
plt.tight_layout(pad=0.5)  
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'RPhi_xt.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'RPhi_xt_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 5))  
plt.pcolormesh(xm, tm, RP_t[:nt_data,:], vmin=-RP_lim, vmax=RP_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('Diamagnetic Reynolds Stress: $R_d$')
plt.colorbar()
plt.tight_layout(pad=0.5) 
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'RP_xt.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'RP_xt_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 5))  
plt.pcolormesh(xm, tm, R_t[:nt_data,:], vmin=-R_lim, vmax=R_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('Reynolds Stress: $R=R_\\phi + R_d$')
plt.colorbar()
plt.tight_layout(pad=0.5) 
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'R_xt.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'R_xt_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 5))  
plt.pcolormesh(xm, tm, RP_frac_t[:nt_data,:], vmin=-RP_frac_lim, vmax=RP_frac_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('$R_d / (R_\\phi + R_d)$')
plt.colorbar()
plt.tight_layout(pad=0.5) 
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'RP_frac_xt.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'RP_frac_xt_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 5))  
plt.pcolormesh(xm, tm, PPhi_t[:nt_data,:], vmin=-PPhi_lim, vmax=PPhi_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('$R_\\phi$ production: $P_\\phi$')
plt.colorbar()
plt.tight_layout(pad=0.5)  
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'PPhi_xt.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'PPhi_xt_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 5))  
plt.pcolormesh(xm, tm, PP_t[:nt_data,:], vmin=-PP_lim, vmax=PP_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('$R_d$ production: $P_d$')
plt.colorbar()
plt.tight_layout(pad=0.5) 
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'PP_xt.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'PP_xt_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 5))  
plt.pcolormesh(xm, tm, P_t[:nt_data,:], vmin=-P_lim, vmax=P_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('$R$ production: $P=P_\\phi + P_d$')
plt.colorbar()
plt.tight_layout(pad=0.5) 
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'P_xt.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'P_xt_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 5))  
plt.pcolormesh(xm, tm, PP_frac_t[:nt_data,:], vmin=-PP_frac_lim, vmax=PP_frac_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('$P_d / (P_\\phi + P_d)$')
plt.colorbar()
plt.tight_layout(pad=0.5) 
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'PP_frac_xt.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'PP_frac_xt_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

Q_lim = np.mean(np.abs(Q_t))
plt.figure(figsize=(6, 5))  
plt.pcolormesh(xm, tm, Q_t[:nt_data,:], vmin=-Q_lim, vmax=Q_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('$\\gamma t$')
plt.title('Heat flux: $Q$')
plt.colorbar()
plt.tight_layout(pad=0.5)  
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'Q_xt.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'Q_xt_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
# plt.show()