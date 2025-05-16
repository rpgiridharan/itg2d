#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from src.mlsarray import mlsarray,slicelist,irft2, rft2, irft,rft,init_kspace_grid
import os
from functools import partial

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2  

#%% Load the HDF5 file
datadir = 'data/'
file_name = datadir+'out_kapt_1_2_chi_0_1_D_0_0_e0_H_1_0_e3.h5'
# file_name = datadir+'out_3f_kapt_1_2_chi_0_1_kz_0_1.h5'
it = -1
# it=100
with h5.File(file_name, 'r', swmr=True) as fl:
    Om = np.mean(fl['fields/Om'][-400:],axis=0)
    P = np.mean(fl['fields/P'][-400:],axis=0)
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
sl=slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
print('kx shape', kx.shape)
nt = len(t)
print("nt: ", nt)

#%% Functions for energy and enstrophy

def ES(omk, kp, k, dk):
    ''' Returns the kinetic energy spectrum'''
    ek = np.abs(omk)**2/kp**2

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(kp>k[i]-dk/2,kp<k[i]+dk/2))])
    return Ek

def ES_ZF(omk, kp, k, dk, slbar):
    ''' Returns the zonal kinetic energy spectrum'''   
    ek_ZF = np.abs(omk[slbar])**2/kp[slbar]**2
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])
    return Ek_ZF

def WS(omk, kp, k , dk):
    ''' Returns the enstrophy spectrum'''    
    wk = np.abs(omk)**2 

    Wk = np.zeros(len(k))
    for i in range(len(k)):
        Wk[i] = np.sum(wk[np.where(np.logical_and(kp>=k[i]-dk/2, kp<k[i]+dk/2))])
    return Wk
    
def WS_ZF(omk, kp, k, dk, slbar):
    ''' Returns the zonal enstrophy spectrum'''    
    wk_ZF = np.abs(omk[slbar])**2

    Wk_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Wk_ZF[i] = np.sum(wk_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])
    return Wk_ZF

def PS(tk, kp, k, dk):
    ''' Returns the pressure spectrum'''
    tk = np.abs(tk)
    Tk = np.zeros(len(k))
    for i in range(len(k)):
        Tk[i] = np.sum(tk[np.where(np.logical_and(kp>=k[i]-dk/2, kp<k[i]+dk/2))])
    return Tk

def PS_ZF(tk, kp, k, dk, slbar):
    ''' Returns the zonal pressure spectrum'''   
    tk_ZF = np.abs(tk[slbar])
    
    Tk_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Tk_ZF[i] = np.sum(tk_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])
    return Tk_ZF

# def rft2(u):
#     Npx = u.shape[-2]
#     Nx, Ny = int(Npx/3)*2, int(Npx/3)*2
#     Nxh = int(Nx/2)
#     yk= np.fft.rfft2(u, norm='forward', axes=(-2,-1))
    
#     if len(u.shape)==2:
#         uk = np.zeros((Nx, int(Ny/2)+1), dtype=complex)
#         uk[:Nxh,:-1] = yk[:Nxh,:int(Ny/2)]
#         uk[-1:-Nxh:-1,:-1] = yk[-1:-Nxh:-1,:int(Ny/2)]
#         uk[0, 0] = 0

#     else:
#         uk = np.zeros((u.shape[0], Nx, int(Ny/2)+1), dtype=complex)
#         uk[:, :Nxh,:-1] = yk[:, :Nxh,:int(Ny/2)]
#         uk[:, -1:-Nxh:-1,:-1] = yk[:, -1:-Nxh:-1,:int(Ny/2)]
#         uk[:, 0, 0] = 0
        
#     return np.hstack(uk)
#%% Plots

Omk=cp.asnumpy(rft2(cp.asarray(Om), sl))
Pk=cp.asnumpy(rft2(cp.asarray(P), sl))

# Omk=rft2(Om)
# Pk=rft2(P)
print(Omk.shape)

dk = ky[1]-ky[0]
kp = np.sqrt(np.abs(kx)**2 + np.abs(ky)**2)
# k = np.logspace(np.log10(np.min(kp)), np.log10(np.max(kp)), num=int(np.max(kp)/dk))
k = np.linspace(np.min(kp), np.max(kp), num=int(np.max(kp)/dk))

Ek = ES(Omk, kp, k, dk)
Ek_ZF = ES_ZF(Omk, kp, k, dk, slbar)
Ek_turb = Ek-Ek_ZF
plt.figure()
plt.loglog(k[1:-1], Ek[1:-1], label = '$E_{k,total}$')
plt.loglog(k[Ek_ZF>0][1:-1], Ek_ZF[Ek_ZF>0][1:-1], label = '$E_{k,ZF}$')
plt.loglog(k[1:-1], Ek_turb[1:-1], label = '$\\mathcal{E}_{k,turb}$')
plt.loglog(k[1:-1], k[1:-1]**(-3), 'k--', label = '$k^{-3}$')
plt.xlabel('$k$')
plt.ylabel('$\\mathcal{E}_k$')
plt.title('$\\mathcal{E}_k(k)$; $t = %.1f$' %t[it])
plt.legend()
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
plt.loglog(k[1:-1], k[1:-1]**(-1), 'k--', label = '$k^{-1}$')
plt.xlabel('$k$')
plt.ylabel('$\\mathcal{W}_k$')
plt.title('$\\mathcal{W}_k(k)$; $t = %.1f$' %t[it])
plt.legend()
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
plt.loglog(k[1:-1], Pkp[1:-1], label = '$\\mathcal{P}_{k,total}$')
plt.loglog(k[Pkp_ZF>0][1:-1], Pkp_ZF[Pkp_ZF>0][1:-1], label = '$\\mathcal{P}_{k,ZF}$')
plt.loglog(k[1:-1], Pkp_turb[1:-1], label = '$\\mathcal{P}_{k,turb}$')
plt.loglog(k[1:-1], k[1:-1]**(-2), 'k--', label = '$k^{-2}$')
plt.xlabel('$k$')
plt.ylabel('$\\mathcal{P}_k$')
plt.title('$\\mathcal{P}_k(k)$; $t = %.1f$' %t[it])
plt.legend()
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'pressure_spectrum.png', dpi=600)
else:
    plt.savefig(datadir+"pressure_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

# %%
