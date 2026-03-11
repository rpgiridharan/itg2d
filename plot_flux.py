#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np
from modules.plot_basics import symmetrize_y_axis
import os

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
datadir = 'data/'
fname = datadir+'out_kapt_1_2_chi_0_1_D_0_1_H_1_0_em3.h5'
it = -1
with h5.File(fname, 'r', swmr=True) as fl:
    Om = np.mean(fl['fields/Om'][-200:],0)
    R = np.mean(fl['fluxes/R'][-200:],0)
    PiP = np.mean(fl['fluxes/PiP'][-200:],0)
    Q = np.mean(fl['fluxes/Q'][-200:],0)
    vbar = np.mean(fl['zonal/vbar'][-200:],0)
    Ombar = np.mean(fl['zonal/Ombar'][-200:],0)
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

plt.figure(figsize=(16, 9))  # Set specific figure size
plt.plot(x[:,0],R,label='$\\Pi_{\\mathrm{\\phi}}$')
plt.plot(x[:,0],PiP,label='$\\Pi_{\\mathrm{P}}$')
plt.plot(x[:,0],R+PiP,label='$\\Pi_{\\mathrm{t}}=\\Pi_{\\mathrm{P}}+\\Pi_{\\mathrm{\\phi}}$')
ylim = plt.gca().get_ylim()
plt.plot(x[:,0],0.5*ylim[-1]*Ombar/np.max(np.abs(Ombar)),'k',label='$\\partial_x^2\\overline{\\phi}$')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('x')
plt.title('$\\Pi(x)$' %t[it])
plt.legend()
symmetrize_y_axis(plt.gca())
plt.tight_layout(pad=0.5)  # Reduce padding
if fname.endswith('out.h5'):
    plt.savefig(datadir+'Pi.png', dpi=100, bbox_inches='tight')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'Pi_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')

plt.figure(figsize=(16, 9))  # Set specific figure size
plt.plot(x[:,0],Q,label='$Q$')
plt.plot(x[:,0],np.mean(Q)+0.05*np.max(np.abs(Q))*vbar/np.max(np.abs(vbar)),'k',label='$\\overline{v}_y$')
# plt.plot(x[:,0],np.mean(Q)+0.05*np.max(np.abs(Q))*Ombar/np.max(Ombar),label='$\\partial_x\\overline{v}_y$')
plt.axhline(np.mean(Q), color='gray', linestyle='--')
plt.xlabel('x')
plt.ylabel('$Q(x)$')
plt.title('$Q(x)$' %t[it])
plt.legend()
# symmetrize_y_axis(plt.gca())
plt.tight_layout(pad=0.5)  # Reduce padding
if fname.endswith('out.h5'):
    plt.savefig(datadir+'Q.png', dpi=100, bbox_inches='tight')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'Q_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

# %%
