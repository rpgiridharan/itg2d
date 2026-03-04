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

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
datadir = 'data_2d3c/'
fname = datadir+'out_2d3c_kapt_2_0_D_0_05_kz_0_0127.h5'
with h5.File(fname, 'r', swmr=True) as fl:
    t = fl['fields/t'][:]
    nt = len(t)
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]
    kapn = fl['params/kapn'][()]
    kapt = fl['params/kapt'][()]
    kapb = fl['params/kapb'][()]
    if 'D' in fl['params']:
        D = fl['params/D'][()]
    elif 'chi' in fl['params']:
        chi = fl['params/chi'][()]
        D = chi
    kz = fl['params/kz'][()]

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
gammax=gam_max(kx,ky,kapn,kapt,kapb,D,kz)
t=t*gammax

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

#%% compute quantities


dk = ky[1]-ky[0]
kp = np.sqrt(np.abs(kx)**2 + np.abs(ky)**2)
# k = np.linspace(np.min(kp), np.max(kp), num=int(np.max(kp)/dk))
k = np.linspace(np.min(ky), np.max(ky), num=int(np.max(ky)/dk))

# MPI parallelization for time series calculation
nt2 = int(nt/2)
nt2 = nt2 - (nt2 % size)
if rank == 0:
    indices = np.array_split(range(nt2), size)
else:
    indices = None
local_indices = comm.scatter(indices, root=0)

# Local arrays for each process
P2k_local = np.zeros((len(local_indices), len(k)))
P2k_ZF_local = np.zeros((len(local_indices), len(k)))
V2k_local = np.zeros((len(local_indices), len(k)))
V2k_ZF_local = np.zeros((len(local_indices), len(k)))
Ek_local = np.zeros((len(local_indices), len(k)))
Ek_ZF_local = np.zeros((len(local_indices), len(k)))
Kk_local = np.zeros((len(local_indices), len(k)))
Kk_ZF_local = np.zeros((len(local_indices), len(k)))
Wk_local = np.zeros((len(local_indices), len(k)))
Wk_ZF_local = np.zeros((len(local_indices), len(k)))
Gk_local = np.zeros((len(local_indices), len(k)))
Gk_ZF_local = np.zeros((len(local_indices), len(k)))
GKk_local = np.zeros((len(local_indices), len(k)))
GKk_ZF_local = np.zeros((len(local_indices), len(k)))

with h5.File(fname, 'r', swmr=True) as fl:
    for idx, it in enumerate(local_indices):
        Omk = fl['fields/Omk'][it+nt//2]
        Pk = fl['fields/Pk'][it+nt//2]
        Vk = fl['fields/Vk'][it+nt//2]
        P2k_local[idx,:] = PS(Pk, kp, k, dk)
        P2k_ZF_local[idx,:] = PS_ZF(Pk, kp, k, dk, slbar)
        V2k_local[idx,:] = VS(Vk, kp, k, dk)
        V2k_ZF_local[idx,:] = VS_ZF(Vk, kp, k, dk, slbar)
        Ek_local[idx,:] = ES(Omk, kp, k, dk)
        Ek_ZF_local[idx,:] = ES_ZF(Omk, kp, k, dk, slbar)
        Kk_local[idx,:] = KS(Omk, kp, k, dk)
        Kk_ZF_local[idx,:] = KS_ZF(Omk, kp, k, dk, slbar)
        Wk_local[idx,:] = WS(Omk, kp, k, dk)
        Wk_ZF_local[idx,:] = WS_ZF(Omk, kp, k, dk, slbar)
        Gk_local[idx,:] = GS(Omk, Pk, kp, k, dk)
        Gk_ZF_local[idx,:] = GS_ZF(Omk, Pk, kp, k, dk, slbar)
        GKk_local[idx,:] = GKS(Omk, Pk, kp, k, dk)
        GKk_ZF_local[idx,:] = GKS_ZF(Omk, Pk, kp, k, dk, slbar)

# Gather results from all processes
P2k_t = P2k_ZF_t = V2k_t = V2k_ZF_t = Ek_t = Ek_ZF_t = Kk_t = Kk_ZF_t = Wk_t = Wk_ZF_t = Gk_t = Gk_ZF_t = GKk_t = GKk_ZF_t = None
if rank == 0:
    P2k_t = np.zeros((nt2, len(k)))
    P2k_ZF_t = np.zeros((nt2, len(k)))
    V2k_t = np.zeros((nt2, len(k)))
    V2k_ZF_t = np.zeros((nt2, len(k)))
    Ek_t = np.zeros((nt2, len(k)))
    Ek_ZF_t = np.zeros((nt2, len(k)))
    Kk_t = np.zeros((nt2, len(k)))
    Kk_ZF_t = np.zeros((nt2, len(k)))
    Wk_t = np.zeros((nt2, len(k)))
    Wk_ZF_t = np.zeros((nt2, len(k)))
    Gk_t = np.zeros((nt2, len(k)))
    Gk_ZF_t = np.zeros((nt2, len(k)))
    GKk_t = np.zeros((nt2, len(k)))
    GKk_ZF_t = np.zeros((nt2, len(k)))

comm.Gather(P2k_local, P2k_t, root=0)
comm.Gather(P2k_ZF_local, P2k_ZF_t, root=0)
comm.Gather(V2k_local, V2k_t, root=0)
comm.Gather(V2k_ZF_local, V2k_ZF_t, root=0)
comm.Gather(Ek_local, Ek_t, root=0)
comm.Gather(Ek_ZF_local, Ek_ZF_t, root=0)
comm.Gather(Kk_local, Kk_t, root=0)
comm.Gather(Kk_ZF_local, Kk_ZF_t, root=0)
comm.Gather(Wk_local, Wk_t, root=0)
comm.Gather(Wk_ZF_local, Wk_ZF_t, root=0)
comm.Gather(Gk_local, Gk_t, root=0)
comm.Gather(Gk_ZF_local, Gk_ZF_t, root=0)
comm.Gather(GKk_local, GKk_t, root=0)
comm.Gather(GKk_ZF_local, GKk_ZF_t, root=0)

if rank == 0:
    print("Gathered")
    P2k_turb_t = P2k_t - P2k_ZF_t
    V2k_turb_t = V2k_t - V2k_ZF_t
    Ek_turb_t = Ek_t - Ek_ZF_t
    Kk_turb_t = Kk_t - Kk_ZF_t
    Wk_turb_t = Wk_t - Wk_ZF_t
    Gk_turb_t = Gk_t - Gk_ZF_t
    GKk_turb_t = GKk_t - GKk_ZF_t

    P2k = np.mean(P2k_t, axis=0)
    P2k_ZF = np.mean(P2k_ZF_t, axis=0)
    P2k_turb = P2k - P2k_ZF
    V2k = np.mean(V2k_t, axis=0)
    V2k_ZF = np.mean(V2k_ZF_t, axis=0)
    V2k_turb = V2k - V2k_ZF
    Ek = np.mean(Ek_t, axis=0)
    Ek_ZF = np.mean(Ek_ZF_t, axis=0)
    Ek_turb = Ek - Ek_ZF
    Kk = np.mean(Kk_t, axis=0)
    Kk_ZF = np.mean(Kk_ZF_t, axis=0)
    Kk_turb = Kk - Kk_ZF
    Wk = np.mean(Wk_t, axis=0)
    Wk_ZF = np.mean(Wk_ZF_t, axis=0)
    Wk_turb = Wk - Wk_ZF
    Gk = np.mean(Gk_t, axis=0)
    Gk_ZF = np.mean(Gk_ZF_t, axis=0)
    Gk_turb = Gk - Gk_ZF
    GKk = np.mean(GKk_t, axis=0)
    GKk_ZF = np.mean(GKk_ZF_t, axis=0)
    GKk_turb = GKk - GKk_ZF

#%% Plots

if rank == 0:
    Pkp = PS(Pk, kp, k, dk)
    Pkp_ZF = PS_ZF(Pk, kp, k, dk, slbar)
    Pkp_turb = Pkp-Pkp_ZF
    plt.figure()
    plt.loglog(k[1:-1], Pkp[1:-1], label = '$P_{k}^2$')
    plt.loglog(k[1:-1], Pkp_ZF[1:-1], label = '$P_{k,ZF}^2$')
    plt.loglog(k[1:-1], Pkp_turb[1:-1], label = '$P_{k,turb}^2$')
    plt.loglog(k[1:-1], k[1:-1]**(-3), 'k--', label = '$k^{-3}$')
    plt.loglog(k[1:-1], k[1:-1]**(-4), 'r--', label = '$k^{-4}$')
    plt.xlabel('$k$')
    plt.ylabel('$P_k^2$')
    plt.title('$P_k^2$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if fname.endswith('out.h5'):
        plt.savefig(datadir+'pressure_spectrum.pdf')
    else:
        plt.savefig(datadir+"pressure_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'))
    plt.show()

    Vkp = VS(Vk, kp, k, dk)
    Vkp_ZF = VS_ZF(Vk, kp, k, dk, slbar)
    Vkp_turb = Vkp-Vkp_ZF
    plt.figure()
    plt.loglog(k[1:-1], Vkp[1:-1], label = '$V_{k}^2$')
    plt.loglog(k[1:-1], Vkp_ZF[1:-1], label = '$V_{k,ZF}^2$')
    plt.loglog(k[1:-1], Vkp_turb[1:-1], label = '$V_{k,turb}^2$')
    plt.loglog(k[1:-1], k[1:-1]**(-2), 'k--', label = '$k^{-2}$')
    plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
    plt.xlabel('$k$')
    plt.ylabel('$V_k^2$')
    plt.title('$V_k^2$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if fname.endswith('out.h5'):
        plt.savefig(datadir+'parallel_velocity_spectrum.pdf')
    else:
        plt.savefig(datadir+"parallel_velocity_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'))
    plt.show()

    Ek = ES(Omk, kp, k, dk)
    Ek_ZF = ES_ZF(Omk, kp, k, dk, slbar)
    Ek_turb = Ek-Ek_ZF
    plt.figure()
    plt.loglog(k[1:-1], Ek[1:-1], label = '$\\mathcal{E}_{k}$')
    plt.loglog(k[1:-1], Ek_ZF[1:-1], label = '$\\mathcal{E}_{k,ZF}$')
    plt.loglog(k[1:-1], Ek_turb[1:-1], label = '$\\mathcal{E}_{k,turb}$')
    plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
    plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
    plt.xlabel('$k$')
    plt.ylabel('$\\mathcal{E}_k$')
    plt.title('$\\mathcal{E}_k$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if fname.endswith('out.h5'):
        plt.savefig(datadir+'energy_spectrum.pdf')
    else:
        plt.savefig(datadir+"energy_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'))
    plt.show()

    Kk = KS(Omk, kp, k, dk)
    Kk_ZF = KS_ZF(Omk, kp, k, dk, slbar)
    Kk_turb = Kk-Kk_ZF
    plt.figure()
    plt.loglog(k[1:-1], Kk[1:-1], label = '$\\mathcal{E}_{kin,k}$')
    plt.loglog(k[1:-1], Kk_ZF[1:-1], label = '$\\mathcal{E}_{kin,k,ZF}$')
    plt.loglog(k[1:-1], Kk_turb[1:-1], label = '$\\mathcal{E}_{kin,k,turb}$')
    plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
    plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
    plt.xlabel('$k$')
    plt.ylabel('$\\mathcal{E}_{kin,k}$')
    plt.title('$\\mathcal{E}_{kin,k}$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if fname.endswith('out.h5'):
        plt.savefig(datadir+'kinetic_energy_spectrum.pdf')
    else:
        plt.savefig(datadir+"kinetic_energy_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'))
    plt.show()

    Wk = WS(Omk, kp, k, dk)
    Wk_ZF = WS_ZF(Omk, kp, k, dk, slbar)
    Wk_turb = Wk-Wk_ZF
    plt.figure()
    plt.loglog(k[1:-1], Wk[1:-1], label = '$\\mathcal{W}_{k}$')
    plt.loglog(k[1:-1], Wk_ZF[1:-1], label = '$\\mathcal{W}_{k,ZF}$')
    plt.loglog(k[1:-1], Wk_turb[1:-1], label = '$\\mathcal{W}_{k,turb}$')
    plt.loglog(k[1:-1], k[1:-1]**(1/3), 'k--', label = '$k^{1/3}$')
    plt.loglog(k[1:-1], k[1:-1]**(-1), 'r--', label = '$k^{-1}$')
    plt.xlabel('$k$')
    plt.ylabel('$\\mathcal{W}_k$')
    plt.title('$\\mathcal{W}_k$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if fname.endswith('out.h5'):
        plt.savefig(datadir+'enstrophy_spectrum.pdf')
    else:
        plt.savefig(datadir+"enstrophy_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'))
    plt.show()

    Gk = GS(Omk, Pk, kp, k, dk)
    Gk_ZF = GS_ZF(Omk, Pk, kp, k, dk, slbar)
    Gk_turb = Gk-Gk_ZF
    plt.figure()
    plt.loglog(k[1:-1], Gk[1:-1], label = '$\\mathcal{G}_{k}$')
    plt.loglog(k[1:-1], Gk_ZF[1:-1], label = '$\\mathcal{G}_{k,ZF}$')
    plt.loglog(k[1:-1], Gk_turb[1:-1], label = '$\\mathcal{G}_{k,turb}$')
    plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
    plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
    plt.xlabel('$k$')
    plt.ylabel('$\\mathcal{G}_{k}$')
    plt.title('$\\mathcal{G}_{k}$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if fname.endswith('out.h5'):
        plt.savefig(datadir+'generalized_energy_spectrum.pdf')
    else:
        plt.savefig(datadir+"generalized_energy_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'))
    plt.show()

    GKk = GKS(Omk, Pk, kp, k, dk)
    GKk_ZF = GKS_ZF(Omk, Pk, kp, k, dk, slbar)
    GKk_turb = GKk-GKk_ZF
    plt.figure()
    plt.loglog(k[1:-1], GKk[1:-1], label = '$\\mathcal{G}_{kin,k}$')
    plt.loglog(k[1:-1], GKk_ZF[1:-1], label = '$\\mathcal{G}_{kin,k,ZF}$')
    plt.loglog(k[1:-1], GKk_turb[1:-1], label = '$\\mathcal{G}_{kin,k,turb}$')
    plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
    plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
    plt.xlabel('$k$')
    plt.ylabel('$\\mathcal{G}_{kin,k}$')
    plt.title('$\\mathcal{G}_{kin,k}$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if fname.endswith('out.h5'):
        plt.savefig(datadir+'generalized_kinetic_energy_spectrum.pdf')
    else:
        plt.savefig(datadir+"generalized_kinetic_energy_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'))
    plt.show()

    # Hk = HS(Omk, Vk, kp, k, dk)
    # Hk_ZF = HS_ZF(Omk, Vk, kp, k, dk, slbar)
    # Hk_turb = Hk-Hk_ZF
    # plt.figure()
    # plt.loglog(k[1:-1], Hk[1:-1], label = '$|\\mathcal{H}_{k,total}|$')
    # plt.loglog(k[1:-1], Hk_ZF[1:-1], label = '$|\\mathcal{H}_{k,ZF}|$')
    # plt.loglog(k[1:-1], Hk_turb[1:-1], label = '$|\\mathcal{H}_{k,turb}|$')
    # plt.loglog(k[1:-1], k[1:-1]**(-1), 'k--', label = '$k^{-1}$')
    # plt.xlabel('$k$')
    # plt.ylabel('$|\\mathcal{H}_k|$')
    # plt.title('$|\\mathcal{H}_k|$')
    # plt.legend()
    # plt.grid(which='both', linestyle='--', linewidth=0.5)
    # plt.tight_layout()
    # if fname.endswith('out.h5'):
    #     plt.savefig(datadir+'helicity_spectrum.pdf')
    # else:
    #     plt.savefig(datadir+"helicity_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'))
    # plt.show()


    # %%
