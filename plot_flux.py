#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from modules.mlsarray import mlsarray,slicelist,irft2, rft2
from modules.plot_basics import symmetrize_y_axis, ubar, rft2_g, irft_g
import os

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2  

#%% Load the HDF5 file
datadir = 'data/'
# res="256_256/"
# res = "512_512/"
# res = "512_768/"
res = "768_512/"
# res = "1024_1024/"
if not os.path.exists(datadir + res):
    os.makedirs(datadir + res)
file_name = datadir+res+'out_kapt_1_44_chi_0_22.h5'
it = -1
with h5.File(file_name, 'r', swmr=True) as fl:
    Om = np.mean(fl['fields/Om'][-200:],0)
    R = np.mean(fl['fluxes/R'][-200:],0)
    PiT = np.mean(fl['fluxes/PiT'][-200:],0)
    Q = np.mean(fl['fluxes/Q'][-200:],0)
    vbar = np.mean(fl['fields/zonal/vbar'][-200:],0)
    Ombar = np.mean(fl['fields/zonal/Ombar'][-200:],0)
    t = fl['fluxes/t'][:]
    x = fl['data/x'][:]
    y = fl['data/y'][:]
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]

Npx,Npy=Om.shape[0],Om.shape[1]  
Nx,Ny=2*Npx//3,2*Npy//3  
nt = len(t)
print("nt: ", nt)

sl=slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]

#%% Plots

plt.figure(figsize=(8, 5))  # Set specific figure size
plt.plot(x[:,0],R,label='$\\mathcal{\\Pi}_\\phi$')
plt.plot(x[:,0],PiT,label='$\\mathcal{\\Pi}_T$')
plt.plot(x[:,0],R+PiT,label='$\\Pi_t=\\mathcal{\\Pi}_T+\\mathcal{\\Pi}_\\phi$')
plt.plot(x[:,0],Ombar,'k',label='$\\partial_x^2\\overline{\\phi}$')
# plt.plot(x[:,0],0.25*np.max(np.abs(PiT))*vbar/np.max(vbar),'k',label='$\\overline{v}_y$')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('x')
plt.title('$\\mathcal{\\Pi}_T(x)$' %t[it])
plt.legend()
symmetrize_y_axis(plt.gca())
plt.tight_layout(pad=0.5)  # Reduce padding
if file_name.endswith('out.h5'):
    plt.savefig(datadir+res+'PiT.png', dpi=600, bbox_inches='tight')
else:
    plt.savefig(datadir+res+file_name.split('/')[-1].replace('out_', 'PiT_').replace('.h5', '.png'), dpi=600, bbox_inches='tight')


plt.figure(figsize=(8, 5))  # Set specific figure size
plt.plot(x[:,0],Q,label='$\\mathcal{Q}$')
plt.plot(x[:,0],np.mean(Q)+0.05*np.max(np.abs(Q))*vbar/np.max(vbar),'k',label='$\\overline{v}_y$')
# plt.plot(x[:,0],np.mean(Q)+0.05*np.max(np.abs(Q))*Ombar/np.max(Ombar),label='$\\partial_x\\overline{v}_y$')
plt.axhline(np.mean(Q), color='gray', linestyle='--')
plt.xlabel('x')
plt.ylabel('$\\mathcal{Q}(x)$')
plt.title('$\\mathcal{Q}(x)$' %t[it])
plt.legend()
# symmetrize_y_axis(plt.gca())
plt.tight_layout(pad=0.5)  # Reduce padding
if file_name.endswith('out.h5'):
    plt.savefig(datadir+res+'Q.png', dpi=600, bbox_inches='tight')
else:
    plt.savefig(datadir+res+file_name.split('/')[-1].replace('out_', 'Q_').replace('.h5', '.png'), dpi=600, bbox_inches='tight')
plt.show()

# %%
