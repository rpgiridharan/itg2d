#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np,irftnp,rftnp
from modules.plot_basics import symmetrize_y_axis
import os

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 3  

#%% Load the HDF5 file

datadir = 'data_scan/'
file_name = datadir+'out_kapt_0_8_chi_0_1_D_1_0_em3_H_1_0_em3.h5'

it = -1
with h5.File(file_name, 'r', swmr=True) as fl:
    Q_t = fl['fluxes/Q'][:]
    R_t = fl['fluxes/R'][:]
    PiP_t = fl['fluxes/PiP'][:]
    vbar_t = fl['zonal/vbar'][:]
    Ombar_t = fl['zonal/Ombar'][:]
    t = fl['fluxes/t'][:]
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]

Nx,Ny=2*Npx//3,2*Npy//3  
nt = len(t)  
print("nt: ", nt)  

sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]

xl,yl=np.arange(0,Lx,Lx/Npx),np.arange(0,Ly,Ly/Npy)
x,y=np.meshgrid(np.array(xl),np.array(yl),indexing='ij')

#%% Plots

# Create proper meshgrid for pcolor
nx = x.shape[0]
nt_data = min(nt, vbar_t.shape[0])  # Make sure we don't exceed array bounds

# Create meshgrid with proper dimensions for pcolor
xm, tm = np.meshgrid(x[:, 0], t[:nt_data])

vbar_lim = np.mean(np.abs(vbar_t))
plt.figure(figsize=(6, 5))  # Set specific figure size
plt.pcolormesh(xm, tm, vbar_t[:nt_data,:], vmin=-vbar_lim, vmax=vbar_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Zonal flow: $\\partial_x\\overline{\\phi}$')
plt.colorbar()
plt.tight_layout(pad=0.5)  # Reduce padding
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'vbar_xt.svg', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'vbar_xt_').replace('.h5', '.png'), dpi=600, bbox_inches='tight')
plt.show()

Ombar_lim = np.mean(np.abs(Ombar_t))
plt.figure(figsize=(6, 5))  # Set specific figure size
plt.pcolormesh(xm, tm, Ombar_t[:nt_data,:], vmin=-Ombar_lim, vmax=Ombar_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Zonal shear: $\\partial_x^2\\overline{\\phi}$')
plt.colorbar()
plt.tight_layout(pad=0.5)  # Reduce padding
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'Ombar_xt.svg', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'Ombar_xt_').replace('.h5', '.png'), dpi=600, bbox_inches='tight')
plt.show()

Q_lim = np.mean(np.abs(Q_t))
plt.figure(figsize=(6, 5))  # Set specific figure size
plt.pcolormesh(xm, tm, Q_t[:nt_data,:], vmin=-Q_lim, vmax=Q_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Heat flux: $Q$')
plt.colorbar()
plt.tight_layout(pad=0.5)  # Reduce padding
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'Q_xt.svg', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'Q_xt_').replace('.h5', '.png'), dpi=600, bbox_inches='tight')
plt.show()

R_lim = np.mean(np.abs(R_t))
plt.figure(figsize=(6, 5))  # Set specific figure size
plt.pcolormesh(xm, tm, R_t[:nt_data,:], vmin=-R_lim, vmax=R_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Reynolds Stress: $\\Pi_\\phi$')
plt.colorbar()
plt.tight_layout(pad=0.5)  # Reduce padding
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'R_xt.svg', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'R_xt_').replace('.h5', '.png'), dpi=600, bbox_inches='tight')
plt.show()

PiP_lim = np.mean(np.abs(PiP_t))
plt.figure(figsize=(6, 5))  # Set specific figure size
plt.pcolormesh(xm, tm, PiP_t[:nt_data,:], vmin=-PiP_lim, vmax=PiP_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Diamagnetic Stress: $\\Pi_P$')
plt.colorbar()
plt.tight_layout(pad=0.5)  # Reduce padding
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'PiP_xt.svg', dpi=300, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'PiP_xt_').replace('.h5', '.png'), dpi=600, bbox_inches='tight')
plt.show()
