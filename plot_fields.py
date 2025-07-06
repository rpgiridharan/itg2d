#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
from functools import partial
from modules.mlsarray import irft2 as original_irft2, rft2 as original_rft2, irft as original_irft, rft as original_rft
from modules.mlsarray import Slicelist
import cupy as cp

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 3  
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.width'] = 1.5 
plt.rcParams['ytick.minor.width'] = 1.5 

#%% Load the HDF5 file
datadir = 'data_scan/'
file_name = datadir+'out_kapt_0_8_chi_0_1_D_1_0_em3_H_1_0_em3.h5'

it = -2
with h5.File(file_name, 'r', swmr=True) as fl:
    Omk = fl['fields/Omk'][it]
    Pk = fl['fields/Pk'][it]
    Ombar = fl['zonal/Ombar'][it]
    Pbar = fl['zonal/Pbar'][it]
    vbar = fl['zonal/vbar'][it]
    
    t = fl['fields/t'][:]
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Ny/2)]
nt = len(t)
print("nt: ", nt)

xl,yl=np.linspace(0,Lx,Npx),np.linspace(0,Ly,Npy)
x,y=np.meshgrid(np.array(xl),np.array(yl),indexing='ij')

#%% Plots

irft2 = partial(original_irft2,Npx=Npx,Npy=Npy,Nx=Nx,sl=sl)
rft2 = partial(original_rft2,sl=sl)
irft = partial(original_irft,Npx=Npx,Nx=Nx)
rft = partial(original_rft,Nx=Nx)

Om = irft2(cp.asarray(Omk)).get()
P = irft2(cp.asarray(Pk)).get()

# Plotting function
def plot_colormesh(dat, dat_bar, title, lab_bar, ax):
    c = ax.pcolormesh(x,y,dat, cmap='seismic', vmin=-np.max(np.abs(dat)), vmax=np.max(np.abs(dat)))
    ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*dat_bar/np.max(np.abs(dat_bar)),'w',linewidth=5)
    ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*dat_bar/np.max(np.abs(dat_bar)),'k',label=lab_bar)
    ax.set_title(title)
    ax.set_xlabel('$x$')
    ax.legend(loc='upper right')
    plt.colorbar(c, ax=ax)

# Create subplots for Om and T
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot each dataset
plot_colormesh(Om, vbar, '$\\Omega$', '$\\overline{v}_y$', axs[0])
# plot_colormesh(P, Pbar, '$P$', '$\\overline{P}$', axs[1])
plot_colormesh(P, vbar, '$P$', '$\\overline{v}_y$', axs[1])
axs[0].set_ylabel('$y$')

# Use tight_layout with reduced padding
fig.tight_layout(pad=0.5)

# Add bbox_inches='tight' to savefig calls
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'fields.png', dpi=600, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'fields_').replace('.h5', '.png'), dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot Om
fig, ax = plt.subplots(figsize=(6, 5))
plot_colormesh(Om, vbar, '$\\Omega$', '$\\overline{v}_y$', ax)
ax.set_ylabel('$y$')
fig.tight_layout(pad=0.5)
plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'fields_Om_').replace('.h5', '.png'), dpi=600, bbox_inches='tight')
plt.show()

# # Plot P
# fig, ax = plt.subplots(figsize=(6, 5))
# plot_colormesh(P, Pbar, '$P$', '$\\overline{P}$', ax)
# fig.tight_layout(pad=0.5)
# plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'fields_P_').replace('.h5', '.png'), dpi=600, bbox_inches='tight')
# plt.show()
