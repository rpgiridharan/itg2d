#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np
from modules.plot_basics import symmetrize_y_axis
import os

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2  

#%% Load the HDF5 file
datadir = 'data/'
file_name = datadir+'out_kapt_1_2_chi_0_1_D_1_0_em3_H_1_0_em3.h5'
it = -1
with h5.File(file_name, 'r', swmr=True) as fl:
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

plt.figure(figsize=(8, 5))  # Set specific figure size
plt.plot(x[:,0],R,label='$\\mathcal{\\Pi}_\\phi$')
plt.plot(x[:,0],PiP,label='$\\mathcal{\\Pi}_P$')
plt.plot(x[:,0],R+PiP,label='$\\Pi_t=\\mathcal{\\Pi}_P+\\mathcal{\\Pi}_\\phi$')
ylim = plt.gca().get_ylim()
plt.plot(x[:,0],0.5*ylim[-1]*Ombar/np.max(np.abs(Ombar)),'k',label='$\\partial_x^2\\overline{\\phi}$')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('x')
plt.title('$\\mathcal{\\Pi}(x)$' %t[it])
plt.legend()
symmetrize_y_axis(plt.gca())
plt.tight_layout(pad=0.5)  # Reduce padding
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'Pi.png', dpi=100, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'Pi_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')

plt.figure(figsize=(8, 5))  # Set specific figure size
plt.plot(x[:,0],Q,label='$\\mathcal{Q}$')
plt.plot(x[:,0],np.mean(Q)+0.05*np.max(np.abs(Q))*vbar/np.max(np.abs(vbar)),'k',label='$\\overline{v}_y$')
# plt.plot(x[:,0],np.mean(Q)+0.05*np.max(np.abs(Q))*Ombar/np.max(Ombar),label='$\\partial_x\\overline{v}_y$')
plt.axhline(np.mean(Q), color='gray', linestyle='--')
plt.xlabel('x')
plt.ylabel('$\\mathcal{Q}(x)$')
plt.title('$\\mathcal{Q}(x)$' %t[it])
plt.legend()
# symmetrize_y_axis(plt.gca())
plt.tight_layout(pad=0.5)  # Reduce padding
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'Q.png', dpi=100, bbox_inches='tight')
else:
    plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'Q_').replace('.h5', '.png'), dpi=100, bbox_inches='tight')
plt.show()

# %%
