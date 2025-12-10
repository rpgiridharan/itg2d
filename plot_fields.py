#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
from functools import partial
from modules.mlsarray import irft2np as original_irft2np, rft2np as original_rft2np, irftnp as original_irftnp, rftnp as original_rftnp
from modules.mlsarray import Slicelist
import cupy as cp
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
    'legend.fontsize': 16
})

#%% Load the HDF5 file
datadir = 'data_scan/'
# file_name = datadir + 'out_kapt_0_2_D_0_001_H_7_9_em5_NZ_1024x1024.h5'
# file_name = datadir + 'out_kapt_2_0_D_0_001_H_1_1_em4_NZ_1024x1024.h5'

kapt=0.4
D=0.1
# pattern = datadir + f'out_kapt_{str(kapt).replace(".", "_")}_D_{str(D).replace(".", "_")}*_{Np}x{Np}.h5'
pattern = datadir + f'out_kapt_{str(kapt).replace(".", "_")}_D_{str(D).replace(".", "_")}*.h5'
files = glob.glob(pattern)
if not files:
    print(f"No file found for kappa_T = {kapt}")
else:
    file_name = files[0]

it = -1
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

irft2np = partial(original_irft2np,Npx=Npx,Npy=Npy,Nx=Nx,sl=sl)
rft2np = partial(original_rft2np,sl=sl)
irftnp = partial(original_irftnp,Npx=Npx,Nx=Nx)
rftnp = partial(original_rftnp,Nx=Nx)

Om = irft2np(Omk)
P = irft2np(Pk)

# Plotting function
def plot_colormesh(dat, dat_bar, title, lab_bar, ax):
    c = ax.pcolormesh(x,y,dat, cmap='seismic', vmin=-np.max(np.abs(dat)), vmax=np.max(np.abs(dat)))
    # ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*dat_bar/np.max(np.abs(dat_bar)),'w',linewidth=5)
    # ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*dat_bar/np.max(np.abs(dat_bar)),'k',label=lab_bar)
    ax.set_title(title)
    ax.set_xlabel('$x$')
    # ax.legend(loc='upper right')
    plt.colorbar(c, ax=ax)

# Create subplots for Om and T
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot each dataset
plot_colormesh(Om, vbar, '$\\Omega$', '$\\overline{v}_y$', axs[0])
axs[0].set_ylabel('$y$')

plot_colormesh(P, vbar, '$P$', '$\\overline{v}_y$', axs[1])
fig.tight_layout(pad=0.5) # Use tight_layout with reduced padding

# Add bbox_inches='tight' to savefig calls
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'fields.pdf', dpi=100, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'fields_').replace('.h5', '.pdf'), dpi=100, bbox_inches='tight')
plt.show()

# %%
# Plot Om
fig, ax = plt.subplots(figsize=(6, 5))
c =ax.pcolormesh(x,y,Om, cmap='seismic', vmin=-np.max(np.abs(Om)), vmax=np.max(np.abs(Om)))
ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*vbar/np.max(np.abs(vbar)),'w',linewidth=5)
ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*vbar/np.max(np.abs(vbar)),'k',label='$\\overline{v}_y$')
ax.legend(loc='upper right')
ax.set_title(f'$\\Omega$ for $\\kappa_T={kapt}$')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.colorbar(c, ax=ax)
fig.tight_layout(pad=0.5)
plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'fields_Om_').replace('.h5', '.pdf'), bbox_inches='tight')
plt.show()

# # Plot P
# fig, ax = plt.subplots(figsize=(6, 5))
# plot_colormesh(P, Pbar, '$P$', '$\\overline{P}$', ax)
# fig.tight_layout(pad=0.5)
# plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'fields_P_').replace('.h5', '.pdf'), bbox_inches='tight')
# plt.show()
