#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np,irftnp,rftnp
from modules.gamma_2d3c import gam_max
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
datadir = 'data_2d3c/'
file_name = datadir+'out_2d3c_kapt_1_2_chi_0_1_kz_0_01.h5'
it = -1
# it=100
with h5.File(file_name, 'r', swmr=True) as fl:
    t = fl['fields/t'][:]
    nt = len(t)
    Omk = np.mean(fl['fields/Omk'][-int(nt/2):],axis=0)
    Pk = np.mean(fl['fields/Pk'][-int(nt/2):],axis=0)
    Vk = np.mean(fl['fields/Vk'][-int(nt/2):],axis=0)
    Ombar = np.mean(fl['zonal/Ombar'][-int(nt/2):],axis=0)
    Pbar = np.mean(fl['zonal/Pbar'][-int(nt/2):],axis=0)
    Vbar = np.mean(fl['zonal/Vbar'][-int(nt/2):],axis=0)
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]
    kapn = fl['params/kapn'][()]
    kapt = fl['params/kapt'][()]
    kapb = fl['params/kapb'][()]
    chi = fl['params/chi'][()]
    a = fl['params/a'][()]
    b = fl['params/b'][()]
    s = fl['params/s'][()]
    kz = fl['params/kz'][()]
    HP = fl['params/HP'][()]
    HPhi = fl['params/HPhi'][()]
    HV = fl['params/HV'][()]

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
slky=np.s_[1:int(Ny/2)-1]
gammax=gam_max(kx,ky,kapn,kapt,kapb,chi,a,b,s,kz,HPhi,HP,HV,slky)
t=t*gammax

print('kx shape', kx.shape)
nt = len(t)
print("nt: ", nt)

#%% Functions for energy and enstrophy

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

def VS(vk, kp, k, dk):
    ''' Returns the var(V) spectrum'''
    vk = np.abs(vk)**2

    Vk = np.zeros(len(k))
    for i in range(len(k)):
        Vk[i] = np.sum(vk[np.where(np.logical_and(kp>=k[i]-dk/2, kp<k[i]+dk/2))])*dk
    return Vk

def VS_ZF(vk, kp, k, dk, slbar):
    ''' Returns the zonal var(V) spectrum'''   
    vk_ZF = np.abs(vk[slbar])**2
    
    Vk_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Vk_ZF[i] = np.sum(vk_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])*dk
    return Vk_ZF

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
    ''' Returns the zonal total energy spectrum'''   
    sigk=np.sign(ky[slbar])
    fac = sigk+kp[slbar]**2 
    ek_ZF = 0.5*fac*np.abs(omk[slbar])**2/kp[slbar]**4
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])*dk
    return Ek_ZF

def KS(omk, kp, k, dk):
    ''' Returns the kinetic energy spectrum'''
    ek = np.abs(omk)**2/kp**2

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(kp>k[i]-dk/2,kp<k[i]+dk/2))])*dk
    return Ek

def KS_ZF(omk, kp, k, dk, slbar):
    ''' Returns the zonal kinetic energy spectrum'''  
    ek_ZF = np.abs(omk[slbar])**2/kp[slbar]**2
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])*dk
    return Ek_ZF

def WS(omk, kp, k , dk):
    ''' Returns the enstrophy spectrum'''    
    wk = 0.5*np.abs(omk)**2 
    # wk = 0.5*np.conj(omk)*omk

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

def GS(omk, pk, kp, k, dk):
    ''' Returns the generalized energy spectrum'''
    sigk=np.sign(ky)
    phik=omk/kp**2
    ek = np.abs(sigk*phik+pk)**2+kp**2*np.abs(phik+pk)**2

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(kp>k[i]-dk/2,kp<k[i]+dk/2))])*dk
    return Ek

def GS_ZF(omk, pk, kp, k, dk, slbar):
    ''' Returns the zonal generalized energy spectrum'''  
    sigk=np.sign(ky)
    phik=omk/kp**2
    ek_ZF = np.abs(sigk[slbar]*phik[slbar]+pk[slbar])**2+kp[slbar]**2*np.abs(phik[slbar]+pk[slbar])**2
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])*dk
    return Ek_ZF

def GKS(omk, pk, kp, k, dk):
    ''' Returns the generalized kinetic energy spectrum'''
    sigk=np.sign(ky)
    phik=omk/kp**2
    ek = kp**2*np.abs(phik+pk)**2

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(kp>k[i]-dk/2,kp<k[i]+dk/2))])*dk
    return Ek

def GKS_ZF(omk, pk, kp, k, dk, slbar):
    ''' Returns the zonal generalized kinetic energy spectrum'''  
    sigk=np.sign(ky)
    phik=omk/kp**2
    ek_ZF = kp[slbar]**2*np.abs(phik[slbar]+pk[slbar])**2
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])*dk
    return Ek_ZF

def HS(omk, vk, kp, k, dk):
    '''Returns the kinetic helicity spectrum'''
    hk = 2*np.abs(np.real(np.conj(vk)*omk))

    Hk = np.zeros(len(k))
    for i in range(len(k)):
        Hk[i] = np.sum(hk[np.where(np.logical_and(kp>=k[i]-dk/2, kp<k[i]+dk/2))])*dk
    return Hk

def HS_ZF(omk, vk, kp, k, dk, slbar):
    '''Returns the zonal kinetic helicity spectrum'''
    hk_ZF = 2*np.abs(np.real(np.conj(vk[slbar])*omk[slbar]))

    Hk_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Hk_ZF[i] = np.sum(hk_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])*dk
    return Hk_ZF

#%% Plots

print(Omk.shape)

dk = ky[1]-ky[0]
kp = np.sqrt(np.abs(kx)**2 + np.abs(ky)**2)
# k = np.logspace(np.log10(np.min(kp)), np.log10(np.max(kp)), num=int(np.max(kp)/dk))
k = np.linspace(np.min(kp), np.max(kp), num=int(np.max(kp)/dk))

Pkp = PS(Pk, kp, k, dk)
Pkp_ZF = PS_ZF(Pk, kp, k, dk, slbar)
Pkp_turb = Pkp-Pkp_ZF
plt.figure()
plt.loglog(k[1:-1], Pkp[1:-1], label = '$P_{k}^2$')
plt.loglog(k[Pkp_ZF>0][1:-1], Pkp_ZF[Pkp_ZF>0][1:-1], label = '$P_{k,ZF}^2$')
plt.loglog(k[1:-1], Pkp_turb[1:-1], label = '$P_{k,turb}^2$')
plt.loglog(k[1:-1], k[1:-1]**(-3), 'k--', label = '$k^{-3}$')
plt.loglog(k[1:-1], k[1:-1]**(-4), 'r--', label = '$k^{-4}$')
plt.xlabel('$k$')
plt.ylabel('$P_k^2$')
plt.title('$P_k^2$; $\\gamma t = %.1f$' %t[it])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'pressure_spectrum.png', dpi=600)
else:
    plt.savefig(datadir+"pressure_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

Vkp = VS(Vk, kp, k, dk)
Vkp_ZF = VS_ZF(Vk, kp, k, dk, slbar)
Vkp_turb = Vkp-Vkp_ZF
plt.figure()
plt.loglog(k[1:-1], Vkp[1:-1], label = '$V_{k}^2$')
plt.loglog(k[Vkp_ZF>0][1:-1], Vkp_ZF[Vkp_ZF>0][1:-1], label = '$V_{k,ZF}^2$')
plt.loglog(k[1:-1], Vkp_turb[1:-1], label = '$V_{k,turb}^2$')
plt.loglog(k[1:-1], k[1:-1]**(-2), 'k--', label = '$k^{-2}$')
plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
plt.xlabel('$k$')
plt.ylabel('$V_k^2$')
plt.title('$V_k^2$; $\\gamma t = %.1f$' %t[it])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'parallel_velocity_spectrum.png', dpi=600)
else:
    plt.savefig(datadir+"parallel_velocity_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

Ek = ES(Omk, kp, k, dk)
Ek_ZF = ES_ZF(Omk, kp, k, dk, slbar)
Ek_turb = Ek-Ek_ZF
plt.figure()
plt.loglog(k[1:-1], Ek[1:-1], label = '$\\mathcal{E}_{k}$')
plt.loglog(k[Ek_ZF>0][1:-1], Ek_ZF[Ek_ZF>0][1:-1], label = '$\\mathcal{E}_{k,ZF}$')
plt.loglog(k[1:-1], Ek_turb[1:-1], label = '$\\mathcal{E}_{k,turb}$')
plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
plt.xlabel('$k$')
plt.ylabel('$\\mathcal{E}_k$')
plt.title('$\\mathcal{E}_k$; $\\gamma t = %.1f$' %t[it])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'energy_spectrum.png', dpi=600)
else:
    plt.savefig(datadir+"energy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

Kk = KS(Omk, kp, k, dk)
Kk_ZF = KS_ZF(Omk, kp, k, dk, slbar)
Kk_turb = Kk-Kk_ZF
plt.figure()
plt.loglog(k[1:-1], Kk[1:-1], label = '$\\mathcal{E}_{kin,k}$')
plt.loglog(k[Kk_ZF>0][1:-1], Kk_ZF[Kk_ZF>0][1:-1], label = '$\\mathcal{E}_{kin,k,ZF}$')
plt.loglog(k[1:-1], Kk_turb[1:-1], label = '$\\mathcal{E}_{kin,k,turb}$')
plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
plt.xlabel('$k$')
plt.ylabel('$\\mathcal{E}_{kin,k}$')
plt.title('$\\mathcal{E}_{kin,k}$; $\\gamma t = %.1f$' %t[it])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'kinetic_energy_spectrum.png', dpi=600)
else:
    plt.savefig(datadir+"kinetic_energy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

Wk = WS(Omk, kp, k, dk)
Wk_ZF = WS_ZF(Omk, kp, k, dk, slbar)
Wk_turb = Wk-Wk_ZF
plt.figure()
plt.loglog(k[1:-1], Wk[1:-1], label = '$\\mathcal{W}_{k}$')
plt.loglog(k[Wk_ZF>0][1:-1], Wk_ZF[Wk_ZF>0][1:-1], label = '$\\mathcal{W}_{k,ZF}$')
plt.loglog(k[1:-1], Wk_turb[1:-1], label = '$\\mathcal{W}_{k,turb}$')
plt.loglog(k[1:-1], k[1:-1]**(1/3), 'k--', label = '$k^{1/3}$')
plt.loglog(k[1:-1], k[1:-1]**(-1), 'r--', label = '$k^{-1}$')
plt.xlabel('$k$')
plt.ylabel('$\\mathcal{W}_k$')
plt.title('$\\mathcal{W}_k$; $\\gamma t = %.1f$' %t[it])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'enstrophy_spectrum.png', dpi=600)
else:
    plt.savefig(datadir+"enstrophy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

Gk = GS(Omk, Pk, kp, k, dk)
Gk_ZF = GS_ZF(Omk, Pk, kp, k, dk, slbar)
Gk_turb = Gk-Gk_ZF
plt.figure()
plt.loglog(k[1:-1], Gk[1:-1], label = '$\\mathcal{G}_{k}$')
plt.loglog(k[Gk_ZF>0][1:-1], Gk_ZF[Gk_ZF>0][1:-1], label = '$\\mathcal{G}_{k,ZF}$')
plt.loglog(k[1:-1], Gk_turb[1:-1], label = '$\\mathcal{G}_{k,turb}$')
plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
plt.xlabel('$k$')
plt.ylabel('$\\mathcal{G}_{k}$')
plt.title('$\\mathcal{G}_{k}$; $\\gamma t = %.1f$' %t[it])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'generalized_energy_spectrum.png', dpi=600)
else:
    plt.savefig(datadir+"generalized_energy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

GKk = GKS(Omk, Pk, kp, k, dk)
GKk_ZF = GKS_ZF(Omk, Pk, kp, k, dk, slbar)
GKk_turb = GKk-GKk_ZF
plt.figure()
plt.loglog(k[1:-1], GKk[1:-1], label = '$\\mathcal{G}_{kin,k}$')
plt.loglog(k[GKk_ZF>0][1:-1], GKk_ZF[GKk_ZF>0][1:-1], label = '$\\mathcal{G}_{kin,k,ZF}$')
plt.loglog(k[1:-1], GKk_turb[1:-1], label = '$\\mathcal{G}_{kin,k,turb}$')
plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
plt.xlabel('$k$')
plt.ylabel('$\\mathcal{G}_{kin,k}$')
plt.title('$\\mathcal{G}_{kin,k}$; $\\gamma t = %.1f$' %t[it])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'generalized_kinetic_energy_spectrum.png', dpi=600)
else:
    plt.savefig(datadir+"generalized_kinetic_energy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

# Hk = HS(Omk, Vk, kp, k, dk)
# Hk_ZF = HS_ZF(Omk, Vk, kp, k, dk, slbar)
# Hk_turb = Hk-Hk_ZF
# plt.figure()
# plt.loglog(k[1:-1], Hk[1:-1], label = '$|\\mathcal{H}_{k,total}|$')
# plt.loglog(k[Hk_ZF>0][1:-1], Hk_ZF[Hk_ZF>0][1:-1], label = '$|\\mathcal{H}_{k,ZF}|$')
# plt.loglog(k[1:-1], Hk_turb[1:-1], label = '$|\\mathcal{H}_{k,turb}|$')
# plt.loglog(k[1:-1], k[1:-1]**(-1), 'k--', label = '$k^{-1}$')
# plt.xlabel('$k$')
# plt.ylabel('$|\\mathcal{H}_k|$')
# plt.title('$|\\mathcal{H}_k|$; $\\gamma t = %.1f$' %t[it])
# plt.legend()
# plt.grid(which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# if file_name.endswith('out.h5'):
#     plt.savefig(datadir+'helicity_spectrum.png', dpi=600)
# else:
#     plt.savefig(datadir+"helicity_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
# plt.show()


# %%
