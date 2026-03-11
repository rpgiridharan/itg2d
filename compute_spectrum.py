#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np,irftnp,rftnp
from modules.gamma import gam_max, Dturb   
import os
import glob
from mpi4py import MPI

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

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#%% Load the HDF5 file

# Npx=512
Npx=1024
datadir=f'data/{Npx}/'

# fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

# kapt=2.0
# D=0.1
# pattern = datadir + f'out_kapt_{str(kapt).replace(".", "_")}_D_{str(D).replace(".", "_")}*.h5'
# files = glob.glob(pattern)
# if not files:
#     print(f"No file found for kappa_T = {kapt}")
# else:
#     fname = files[0]

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
    if 'H' in fl['params']:
        H = fl['params/H'][()]
    elif 'HP' in fl['params']:
        HP = fl['params/HP'][()]
        H = HP

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
gammax=gam_max(kx,ky,kapn,kapt,kapb,D,H)
t=t*gammax

nt = len(t)
if rank == 0:
    print(f"nt: {nt}")

#%% Functions for energy and enstrophy

def Phi2S(phik, kp, k, dk):
    ''' Returns the var(phi) spectrum'''
    phi2k = np.abs(phik)**2
    
    Phi2k = np.zeros(len(k))
    for i in range(len(k)):
        Phi2k[i] = np.sum(phi2k[np.where(np.logical_and(kp>=k[i]-dk/2, kp<k[i]+dk/2))])*dk
    return Phi2k

def Phi2S_ZF(phik, kp, k, dk, slbar):
    ''' Returns the zonal var(phi) spectrum'''   
    phi2k_ZF = np.abs(phik[slbar])**2
    
    Phi2k_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Phi2k_ZF[i] = np.sum(phi2k_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])*dk
    return Phi2k_ZF

def P2S(pk, kp, k, dk):
    ''' Returns the var(P) spectrum'''
    p2k = np.abs(pk)**2
    
    P2k = np.zeros(len(k))
    for i in range(len(k)):
        P2k[i] = np.sum(p2k[np.where(np.logical_and(kp>=k[i]-dk/2, kp<k[i]+dk/2))])*dk
    return P2k

def P2S_ZF(pk, kp, k, dk, slbar):
    ''' Returns the zonal var(P) spectrum'''   
    pk_ZF = np.abs(pk[slbar])**2
    
    P2k_ZF = np.zeros(len(k))
    for i in range(len(k)):
        P2k_ZF[i] = np.sum(pk_ZF[np.where(np.logical_and(kp[slbar]>=k[i]-dk/2, kp[slbar]<k[i]+dk/2))])*dk
    return P2k_ZF

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
    phik=omk/kp**2
    ek = kp**2*np.abs(phik+pk)**2

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(kp>k[i]-dk/2,kp<k[i]+dk/2))])*dk
    return Ek

def GKS_ZF(omk, pk, kp, k, dk, slbar):
    ''' Returns the zonal generalized kinetic energy spectrum'''  
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

# Kx,Ky = np.meshgrid(k,k,indexing='ij')
# Dturb=Dturb(Kx,Ky,kapn,kapt,kapb,D,H)

# MPI parallelization for time series calculation
nt2 = int(nt/2)
nt2 = nt2 - (nt2 % size)
if rank == 0:
    indices = np.array_split(range(nt2), size)
else:
    indices = None
local_indices = comm.scatter(indices, root=0)

# Local arrays for each process
Phi2k_local = np.zeros((len(local_indices), len(k)))
Phi2k_ZF_local = np.zeros((len(local_indices), len(k)))
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

with h5.File(fname, 'r', swmr=True) as fl:
    for idx, it in enumerate(local_indices):
        Omk = fl['fields/Omk'][it+nt//2]
        Pk = fl['fields/Pk'][it+nt//2]
        Phi2k_local[idx,:] = Phi2S(Pk, kp, k, dk)
        Phi2k_ZF_local[idx,:] = Phi2S_ZF(Pk, kp, k, dk, slbar)
        P2k_local[idx,:] = P2S(Pk, kp, k, dk)
        P2k_ZF_local[idx,:] = P2S_ZF(Pk, kp, k, dk, slbar)
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
Phi2k_t = Phi2k_ZF_t = P2k_t = P2k_ZF_t = Ek_t = Ek_ZF_t = Kk_t = Kk_ZF_t = Wk_t = Wk_ZF_t = Gk_t = Gk_ZF_t = GKk_t = GKk_ZF_t = None
if rank == 0:
    Phi2k_t = np.zeros((nt2, len(k)))
    Phi2k_ZF_t = np.zeros((nt2, len(k)))
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

comm.Gather(Phi2k_local, Phi2k_t, root=0)
comm.Gather(Phi2k_ZF_local, Phi2k_ZF_t, root=0)
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
    
    Phi2k_turb_t = Phi2k_t - Phi2k_ZF_t
    P2k_turb_t = P2k_t - P2k_ZF_t
    Ek_turb_t = Ek_t - Ek_ZF_t
    Kk_turb_t = Kk_t - Kk_ZF_t
    Wk_turb_t = Wk_t - Wk_ZF_t
    Gk_turb_t = Gk_t - Gk_ZF_t
    GKk_turb_t = GKk_t - GKk_ZF_t

    Phi2k = np.mean(Phi2k_t, axis=0)
    Phi2k_ZF = np.mean(Phi2k_ZF_t, axis=0)
    Phi2k_turb = Phi2k - Phi2k_ZF
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

    #%% Save computed spectra
    savefile = fname.replace('out_', 'spectrum_')
    with h5.File(savefile, 'w') as fl:
        fl.create_dataset('k', data=k)
        fl.create_dataset('Phi2k', data=Phi2k)
        fl.create_dataset('Phi2k_ZF', data=Phi2k_ZF)
        fl.create_dataset('Phi2k_turb', data=Phi2k_turb)
        fl.create_dataset('P2k', data=P2k)
        fl.create_dataset('P2k_ZF', data=P2k_ZF)
        fl.create_dataset('P2k_turb', data=P2k_turb)
        fl.create_dataset('Ek', data=Ek)
        fl.create_dataset('Ek_ZF', data=Ek_ZF)
        fl.create_dataset('Ek_turb', data=Ek_turb)
        fl.create_dataset('Kk', data=Kk)
        fl.create_dataset('Kk_ZF', data=Kk_ZF)
        fl.create_dataset('Kk_turb', data=Kk_turb)
        fl.create_dataset('Wk', data=Wk)
        fl.create_dataset('Wk_ZF', data=Wk_ZF)
        fl.create_dataset('Wk_turb', data=Wk_turb)
        fl.create_dataset('Gk', data=Gk)
        fl.create_dataset('Gk_ZF', data=Gk_ZF)
        fl.create_dataset('Gk_turb', data=Gk_turb)
        fl.create_dataset('GKk', data=GKk)
        fl.create_dataset('GKk_ZF', data=GKk_ZF)
        fl.create_dataset('GKk_turb', data=GKk_turb)
    print(f"Saved spectra to {savefile}")
