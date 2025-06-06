#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
from functools import partial

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2  

#%% Load the HDF5 file
datadir = 'data/'
file_name = datadir+'out_kapt_0_3_chi_0_1_D_1_0_em3_H_1_0_em3.h5'

it = -2
with h5.File(file_name, 'r', swmr=True) as fl:
    Om = fl['fields/Om'][it]
    P = fl['fields/P'][it]
    Ombar = fl['zonal/Ombar'][it][:,0]
    Pbar = fl['zonal/Pbar'][it][:,0]
    vbar = fl['zonal/vbar'][it][:,0]
    
    t = fl['fields/t'][:]
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]

Nx,Ny=2*Npx//3,2*Npy//3  
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Ny/2)]
nt = len(t)
print("nt: ", nt)

xl,yl=np.linspace(0,Lx,Nx-1),np.linspace(0,Ly,Ny-1)
x,y=np.meshgrid(np.array(xl),np.array(yl),indexing='ij')

#%% Plots

# Plotting function
def plot_colormesh(dat, dat_bar, title, lab_bar, ax):
    c = ax.pcolormesh(x,y,dat, cmap='seismic', vmin=-np.max(np.abs(dat)), vmax=np.max(np.abs(dat)))
    # ax.plot(x[:,0],0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*vbar/np.max(np.abs(vbar)),'k',label='$\\overline{v}_y$')
    ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*dat_bar/np.max(np.abs(dat_bar)),'k',label=lab_bar)
    ax.set_title(title)
    ax.set_xlabel('$x$')
    ax.legend(loc='upper right')
    plt.colorbar(c, ax=ax)

# Create subplots for Om and T
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot each dataset
plot_colormesh(Om, vbar, '$\\Omega$', '$\\overline{v}_y$', axs[0])
plot_colormesh(P, Pbar, '$P$', '$\\overline{P}$', axs[1])
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
# # Plot Om
# fig, ax = plt.subplots(figsize=(6, 5))
# plot_colormesh(Om, Ombar, '$\\Omega$', '$\\overline{\\Omega}$', ax)
# ax.set_ylabel('$y$')
# fig.tight_layout(pad=0.5)
# plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'fields_Om_').replace('.h5', '.png'), dpi=600, bbox_inches='tight')
# plt.show()

# # Plot T
# fig, ax = plt.subplots(figsize=(6, 5))
# plot_colormesh(P, Pbar, '$P$', '$\\overline{P}$', ax)
# fig.tight_layout(pad=0.5)
# plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'fields_P_').replace('.h5', '.png'), dpi=600, bbox_inches='tight')
# plt.show()
