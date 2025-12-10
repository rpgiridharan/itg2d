#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np,irftnp,rftnp
from modules.gamma import gam_max   
import os
import glob
from mpi4py import MPI

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 3  
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.width'] = 1.5 
plt.rcParams['ytick.minor.width'] = 1.5 

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#%% Load the HDF5 file
datadir = 'data/'
# file_name = datadir + 'out_kapt_2_0_D_0_01_H_0_0_e0_NZ_1024x1024.h5'
file_name = datadir + 'out_kapt_2_0_D_0_01_H_0_0_e0_NZ_1024x1024.h5'

# kapt=2.0
# D=1e-3
# Np=1024
# pattern = datadir + f'out_kapt_{str(kapt).replace(".", "_")}_D_{str(D).replace(".", "_")}*_{Np}x{Np}.h5'
# files = glob.glob(pattern)
# if not files:
#     print(f"No file found for kappa_T = {kapt}")
# else:
#     file_name = files[0]

with h5.File(file_name, 'r', swmr=True) as fl:
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
    HP = fl['params/HP'][()]
    HPhi = fl['params/HPhi'][()]

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
gammax=gam_max(kx,ky,kapn,kapt,kapb,D,HP,HPhi)
t=t*gammax

nt = len(t)
print(f"nt: {nt}")

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

def ES(omk, kp, k, dk):
    ''' Returns the total energy spectrum'''
    sigk=np.sign(ky)
    fac = sigk+kp**2
    ek = fac*np.abs(omk)**2/kp**4

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(kp>k[i]-dk/2,kp<k[i]+dk/2))])*dk
    return Ek

def ES_ZF(omk, kp, k, dk, slbar):
    ''' Returns the zonal total energy spectrum'''  
    sigk=np.sign(ky[slbar])
    fac = sigk+kp[slbar]**2 
    ek_ZF = fac*np.abs(omk[slbar])**2/kp[slbar]**4
    
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
    wk = np.abs(omk)**2 

    Wk = np.zeros(len(k))
    for i in range(len(k)):
        Wk[i] = np.sum(wk[np.where(np.logical_and(kp>=k[i]-dk/2, kp<k[i]+dk/2))])*dk
    return Wk
    
def WS_ZF(omk, kp, k, dk, slbar):
    ''' Returns the zonal enstrophy spectrum'''    
    wk_ZF = np.abs(omk[slbar])**2

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

#%% compute quantities

dk = ky[1]-ky[0]
kp = np.sqrt(np.abs(kx)**2 + np.abs(ky)**2)
# k = np.linspace(np.min(kp), np.max(kp), num=int(np.max(kp)/dk))
k = np.linspace(np.min(ky), np.max(ky), num=int(np.max(ky)/dk))

# MPI parallelization for time series calculation
nstart=55
nt2 = len(t)-nstart
nt2 = nt2 - (nt2 % size)
if rank == 0:
    indices = np.array_split(range(nt2), size)
else:
    indices = None

local_indices = comm.scatter(indices, root=0)

# Local arrays for each process
P2k_local = np.zeros((len(local_indices), len(k)))
P2k_ZF_local = np.zeros((len(local_indices), len(k)))
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

with h5.File(file_name, 'r', swmr=True) as fl:
    for idx, it in enumerate(local_indices):
        Omk = fl['fields/Omk'][it+nstart]
        Pk = fl['fields/Pk'][it+nstart]
        P2k_local[idx,:] = PS(Pk, kp, k, dk)
        P2k_ZF_local[idx,:] = PS_ZF(Pk, kp, k, dk, slbar)
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
P2k_t = P2k_ZF_t = Ek_t = Ek_ZF_t = Kk_t = Kk_ZF_t = Wk_t = Wk_ZF_t = Gk_t = Gk_ZF_t = GKk_t = GKk_ZF_t = None
if rank == 0:
    P2k_t = np.zeros((nt2, len(k)))
    P2k_ZF_t = np.zeros((nt2, len(k)))
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
    Ek_turb_t = Ek_t - Ek_ZF_t
    Kk_turb_t = Kk_t - Kk_ZF_t
    Wk_turb_t = Wk_t - Wk_ZF_t
    Gk_turb_t = Gk_t - Gk_ZF_t
    GKk_turb_t = GKk_t - GKk_ZF_t

    P2k = np.mean(P2k_t, axis=0)
    P2k_ZF = np.mean(P2k_ZF_t, axis=0)
    P2k_turb = P2k - P2k_ZF
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
    plt.figure()
    plt.loglog(k[1:-1], P2k[1:-1], label = '$P_{k}^2$')
    plt.loglog(k[P2k_ZF>0][1:-1], P2k_ZF[P2k_ZF>0][1:-1], label = '$P_{k,ZF}^2$')
    plt.loglog(k[1:-1], P2k_turb[1:-1], label = '$P_{k,turb}^2$')
    plt.loglog(k[1:-1], k[1:-1]**(-3), 'k--', label = '$k^{-3}$')
    plt.loglog(k[1:-1], k[1:-1]**(-4), 'r--', label = '$k^{-4}$')
    plt.xlabel('$k$')
    plt.ylabel('$P_k^2$')
    plt.title('$P^2_k$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'pressure_spectrum.pdf', dpi=100)
    else:
        plt.savefig(datadir+"pressure_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
    plt.show()

    plt.figure()
    plt.loglog(k[1:-1], Ek[1:-1], label = '$\\mathcal{E}_{k}$')
    plt.loglog(k[Ek_ZF>0][1:-1], Ek_ZF[Ek_ZF>0][1:-1], label = '$\\mathcal{E}_{k,ZF}$')
    plt.loglog(k[1:-1], Ek_turb[1:-1], label = '$\\mathcal{E}_{k,turb}$')
    plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
    plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
    plt.xlabel('$k$')
    plt.ylabel('$\\mathcal{E}_k$')
    plt.title('$\\mathcal{E}_k$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'energy_spectrum.pdf', dpi=100)
    else:
        plt.savefig(datadir+"energy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
    plt.show()

    plt.figure()
    plt.loglog(k[1:-1], Kk[1:-1], label = '$\\mathcal{E}_{kin,k}$')
    plt.loglog(k[Kk_ZF>0][1:-1], Kk_ZF[Kk_ZF>0][1:-1], label = '$\\mathcal{E}_{kin,k,ZF}$')
    plt.loglog(k[1:-1], Kk_turb[1:-1], label = '$\\mathcal{E}_{kin,k,turb}$')
    plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
    plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
    plt.xlabel('$k$')
    plt.ylabel('$\\mathcal{E}_{kin,k}$')
    plt.title('$\\mathcal{E}_{kin,k}$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'kinetic_energy_spectrum.pdf', dpi=100)
    else:
        plt.savefig(datadir+"kinetic_energy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
    plt.show()

    plt.figure()
    plt.loglog(k[1:-1], Wk[1:-1], label = '$\\mathcal{W}_{k}$')
    plt.loglog(k[Wk_ZF>0][1:-1], Wk_ZF[Wk_ZF>0][1:-1], label = '$\\mathcal{W}_{k,ZF}$')
    plt.loglog(k[1:-1], Wk_turb[1:-1], label = '$\\mathcal{W}_{k,turb}$')
    plt.loglog(k[1:-1], k[1:-1]**(1/3), 'k--', label = '$k^{1/3}$')
    plt.loglog(k[1:-1], k[1:-1]**(-1), 'r--', label = '$k^{-1}$')
    plt.xlabel('$k$')
    plt.ylabel('$\\mathcal{W}_k$')
    plt.title('$\\mathcal{W}_k$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'enstrophy_spectrum.pdf', dpi=100)
    else:
        plt.savefig(datadir+"enstrophy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
    plt.show()

    plt.figure()
    plt.loglog(k[1:-1], Gk[1:-1], label = '$\\mathcal{G}_{k}$')
    plt.loglog(k[Gk_ZF>0][1:-1], Gk_ZF[Gk_ZF>0][1:-1], label = '$\\mathcal{G}_{k,ZF}$')
    plt.loglog(k[1:-1], Gk_turb[1:-1], label = '$\\mathcal{G}_{k,turb}$')
    plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
    plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
    plt.xlabel('$k$')
    plt.ylabel('$\\mathcal{G}_{k}$')
    plt.title('$\\mathcal{G}_{k}$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'generalized_energy_spectrum.pdf', dpi=100)
    else:
        plt.savefig(datadir+"generalized_energy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
    plt.show()

    plt.figure()
    plt.loglog(k[1:-1], GKk[1:-1], label = '$\\mathcal{G}_{kin,k}$')
    plt.loglog(k[GKk_ZF>0][1:-1], GKk_ZF[GKk_ZF>0][1:-1], label = '$\\mathcal{G}_{kin,k,ZF}$')
    plt.loglog(k[1:-1], GKk_turb[1:-1], label = '$\\mathcal{G}_{kin,k,turb}$')
    plt.loglog(k[1:-1], k[1:-1]**(-5/3), 'k--', label = '$k^{-5/3}$')
    plt.loglog(k[1:-1], k[1:-1]**(-3), 'r--', label = '$k^{-3}$')
    plt.xlabel('$k$')
    plt.ylabel('$\\mathcal{G}_{kin,k}$')
    plt.title('$\\mathcal{G}_{kin,k}$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'generalized_kinetic_energy_spectrum.pdf', dpi=100)
    else:
        plt.savefig(datadir+"generalized_kinetic_energy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
    plt.show()

# %%
