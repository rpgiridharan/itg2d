#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np,irftnp,rftnp
import os
from functools import partial

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
datadir = 'data/'
file_name = datadir+'out_kapt_1_2_chi_0_1_H_1_0_em3.h5'
it = -1
# it=100
with h5.File(file_name, 'r', swmr=True) as fl:
    Omk = np.mean(fl['fields/Omk'][-400:],axis=0)
    Pk = np.mean(fl['fields/Pk'][-400:],axis=0)
    Ombar = np.mean(fl['zonal/Ombar'][-400:],axis=0)
    Pbar = np.mean(fl['zonal/Pbar'][-400:],axis=0)
    t = fl['fields/t'][:]
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]


Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
print('kx shape', kx.shape)
nt = len(t)
print("nt: ", nt)

#%% Functions for energy and enstrophy

def ES(omk, kp, k, dk):
    ''' Returns the total energy spectrum'''
    sigk=np.sign(ky)
    fac = sigk+kp**2
    ek = 0.5*fac*np.abs(omk)**2/kp**4

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(kp>k[i]-dk/2,kp<k[i]+dk/2))])*dk
    return Ek

def ES_ZF(omk, kp, k, dk, slbar):
    ''' Returns the zonal kinetic energy spectrum'''  
    sigk=np.sign(ky[slbar])
    fac = sigk+kp[slbar]**2 
    ek_ZF = 0.5*fac*np.abs(omk[slbar])**2/kp[slbar]**4
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])*dk
    return Ek_ZF

def WS(omk, kp, k , dk):
    ''' Returns the enstrophy spectrum'''    
    wk = 0.5*np.abs(omk)**2 

    Wk = np.zeros(len(k))
    for i in range(len(k)):
        Wk[i] = np.sum(wk[np.where(np.logical_and(kp>=k[i]-dk/2, kp<k[i]+dk/2))])*dk
    return Wk
    
def WS_ZF(omk, kp, k, dk, slbar):
    ''' Returns the zonal enstrophy spectrum'''    
    wk_ZF = 0.5*np.abs(omk[slbar])**2

    Wk_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Wk_ZF[i] = np.sum(wk_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])*dk
    return Wk_ZF

def PS(pk, kp, k, dk):
    ''' Returns the var(P) spectrum'''
    pk = np.abs(pk)**2
    
    Pk = np.zeros(len(k))
    for i in range(len(k)):
        Pk[i] = np.sum(pk[np.where(np.logical_and(kp>=k[i]-dk/2, kp<k[i]+dk/2))])*dk
    return Pk

def PS_ZF(pk, kp, k, dk, slbar):
    ''' Returns the zonal var(P) spectrum'''   
    pk_ZF = np.abs(pk[slbar])**2
    
    Pk_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Pk_ZF[i] = np.sum(pk_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])*dk
    return Pk_ZF


#%% Plots

print(Omk.shape)

dk = ky[1]-ky[0]
kp = np.sqrt(np.abs(kx)**2 + np.abs(ky)**2)
# k = np.logspace(np.log10(np.min(kp)), np.log10(np.max(kp)), num=int(np.max(kp)/dk))
k = np.linspace(np.min(kp), np.max(kp), num=int(np.max(kp)/dk))

Ek = ES(Omk, kp, k, dk)
Ek_ZF = ES_ZF(Omk, kp, k, dk, slbar)
Ek_turb = Ek-Ek_ZF
plt.figure()
plt.loglog(k[1:-1], Ek[1:-1], label = '$\\mathcal{E}_{k,total}$')
plt.loglog(k[Ek_ZF>0][1:-1], Ek_ZF[Ek_ZF>0][1:-1], label = '$\\mathcal{E}_{k,ZF}$')
plt.loglog(k[1:-1], Ek_turb[1:-1], label = '$\\mathcal{E}_{k,turb}$')
plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
plt.xlabel('$k$')
plt.ylabel('$\\mathcal{E}_k$')
plt.title('$\\mathcal{E}_k$; $t = %.1f$' %t[it])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'energy_spectrum.png', dpi=600)
else:
    plt.savefig(datadir+"energy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

Wk = WS(Omk, kp, k, dk)
Wk_ZF = WS_ZF(Omk, kp, k, dk, slbar)
Wk_turb = Wk-Wk_ZF
plt.figure()
plt.loglog(k[1:-1], Wk[1:-1], label = '$\\mathcal{W}_{k,total}$')
plt.loglog(k[Wk_ZF>0][1:-1], Wk_ZF[Wk_ZF>0][1:-1], label = '$\\mathcal{W}_{k,ZF}$')
plt.loglog(k[1:-1], Wk_turb[1:-1], label = '$\\mathcal{W}_{k,turb}$')
plt.loglog(k[1:-1], k[1:-1]**(1/3), 'k--', label = '$k^{1/3}$')
plt.loglog(k[1:-1], k[1:-1]**(-1), 'r--', label = '$k^{-1}$')
plt.xlabel('$k$')
plt.ylabel('$\\mathcal{W}_k$')
plt.title('$\\mathcal{W}_k$; $t = %.1f$' %t[it])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'enstrophy_spectrum.png', dpi=600)
else:
    plt.savefig(datadir+"enstrophy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

Pkp = PS(Pk, kp, k, dk)
Pkp_ZF = PS_ZF(Pk, kp, k, dk, slbar)
Pkp_turb = Pkp-Pkp_ZF
plt.figure()
plt.loglog(k[1:-1], Pkp[1:-1], label = '$P_{k,total}^2$')
plt.loglog(k[Pkp_ZF>0][1:-1], Pkp_ZF[Pkp_ZF>0][1:-1], label = '$P_{k,ZF}^2$')
plt.loglog(k[1:-1], Pkp_turb[1:-1], label = '$P_{k,turb}^2$')
plt.loglog(k[1:-1], k[1:-1]**(-3), 'k--', label = '$k^{-3}$')
plt.loglog(k[1:-1], k[1:-1]**(-4), 'r--', label = '$k^{-4}$')
plt.xlabel('$k$')
plt.ylabel('$P_k^2$')
plt.title('$P^2_k$; $t = %.1f$' %t[it])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'pressure_spectrum.png', dpi=600)
else:
    plt.savefig(datadir+"pressure_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

# %%
