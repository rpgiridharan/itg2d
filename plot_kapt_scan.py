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
import glob
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 3  
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.width'] = 1.5 
plt.rcParams['ytick.minor.width'] = 1.5 

#%% Functions for energy, enstrophy and entropy

def E(Omk, ky, kpsq):
    ''' Returns the total energy of the system'''
    sigk=np.sign(ky)
    fac = sigk+kpsq
    Etemp = np.sum(fac*np.abs(Omk)**2/kpsq**2).item()
    return np.real(Etemp)

def E_ZF(Omk, ky, kpsq, slbar):
    ''' Returns the zonal energy of the system'''
    sigk=np.sign(ky)
    fac = sigk+kpsq
    E_ZFtemp = np.sum(fac[slbar]*np.abs(Omk[slbar])**2/kpsq[slbar]**2).item()
    return np.real(E_ZFtemp)

def G(Omk, ky, kpsq):
    ''' Returns the generalized energy of the system'''
    sigk=np.sign(ky)
    Phik = Omk/kpsq
    Gtemp = np.sum(np.abs(sigk*Phik+Pk)**2+kpsq*np.abs(Phik+Pk)**2).item()
    return np.real(Gtemp)

def G_ZF(Omk, ky, kpsq, slbar):
    ''' Returns the zonal generalized energy of the system'''
    sigk=np.sign(ky)
    Phik = Omk/kpsq
    G_ZFtemp = np.sum(np.abs(sigk[slbar]*Phik[slbar]+Pk[slbar])**2+kpsq[slbar]*np.abs(Phik[slbar]+Pk[slbar])**2).item()
    return np.real(G_ZFtemp)

def sigma(Omk, Pk, Q, kx, kpsq, sl):
    ''' Returns the entropy production rate of the system'''
    #-Q/T^2delT/delx
    Phi = irft2np(-Omk/kpsq,Npx,Npy,Nx,sl)
    P = irft2np(Pk,Npx,Npy,Nx,sl)
    T = P/Phi
    Tk = rft2np(T, sl)
    dTdx = irft2np(1j*kx*Tk, sl)
    sig = -Q/T**2 * dTdx
    return sig

#%% Define the quantities to be plotted

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

datadir = 'data_scan/'
kapt_vals=np.arange(0.2,1.0,0.05) 

P2_frac_scan = np.zeros(kapt_vals.shape)
P2_frac_scan_err = np.zeros(kapt_vals.shape)
E_frac_scan = np.zeros(kapt_vals.shape)
E_frac_scan_err = np.zeros(kapt_vals.shape)
G_frac_scan = np.zeros(kapt_vals.shape)
G_frac_scan_err = np.zeros(kapt_vals.shape)
Q_scan = np.zeros(kapt_vals.shape)
Q_scan_err = np.zeros(kapt_vals.shape)


# Parallelize the loop over kapt_vals using MPI
local_indices = np.array_split(np.arange(len(kapt_vals)), size)[rank]

local_P2_frac_scan = np.zeros(kapt_vals.shape)
local_P2_frac_scan_err = np.zeros(kapt_vals.shape)
local_E_frac_scan = np.zeros(kapt_vals.shape)
local_E_frac_scan_err = np.zeros(kapt_vals.shape)
local_G_frac_scan = np.zeros(kapt_vals.shape)
local_G_frac_scan_err = np.zeros(kapt_vals.shape)
local_Q_scan = np.zeros(kapt_vals.shape)
local_Q_scan_err = np.zeros(kapt_vals.shape)

for i in local_indices:
    kapt = round(kapt_vals[i], 3)
    D = 0.1
    print(f'Rank {rank}: Processing kappa_T = {kapt}')
    pattern = datadir + f'out_kapt_{str(kapt).replace(".", "_")}_D_{str(D).replace(".", "_")}*.h5'
    files = glob.glob(pattern)
    if not files:
        print(f"Rank {rank}: No file found for kappa_T = {kapt}")
        continue
    else:
        file_name = files[0]

    with h5.File(file_name, 'r', swmr=True) as fl:
        t = fl['fields/t'][:]
        nt = len(t)
        tf = fl['fluxes/t'][:]
        ntf = len(tf)
        kx = fl['data/kx'][:]
        ky = fl['data/ky'][:]
        Lx = fl['params/Lx'][()]
        Ly = fl['params/Ly'][()]
        Npx= fl['params/Npx'][()]
        Npy= fl['params/Npy'][()]
        Nx,Ny=2*Npx//3,2*Npy//3  
        sl=Slicelist(Nx,Ny)
        slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
        kpsq = kx**2 + ky**2
        P2_frac_t = np.zeros(int(nt/2))
        E_frac_t = np.zeros(int(nt/2))
        G_frac_t = np.zeros(int(nt/2))
        for it in range(int(nt/2)):
            Omk = fl['fields/Omk'][it+nt//2]
            Pk = fl['fields/Pk'][it+nt//2]
            P2_frac_t[it] = np.sum(np.abs(Pk[slbar])**2)/np.sum(np.abs(Pk)**2)
            E_frac_t[it] = E_ZF(Omk, ky, kpsq, slbar) / E(Omk, ky, kpsq)
            G_frac_t[it] = G_ZF(Omk, ky, kpsq, slbar) / G(Omk, ky, kpsq)
        Q_t = np.zeros(int(ntf/2))
        for it in range(int(ntf/2)):
            Q = fl['fluxes/Q'][it+ntf//2]
            Q_t[it] = np.mean(Q)

    local_P2_frac_scan[i] = np.mean(P2_frac_t)
    local_E_frac_scan[i]= np.mean(E_frac_t)
    local_G_frac_scan[i] = np.mean(G_frac_t)
    local_Q_scan[i] = np.mean(Q_t)

    local_P2_frac_scan_err[i] = np.std(P2_frac_t)
    local_E_frac_scan_err[i] = np.std(E_frac_t)
    local_G_frac_scan_err[i] = np.std(G_frac_t)
    local_Q_scan_err[i] = np.std(Q_t)

# Gather results from all ranks to rank 0
comm.Reduce(local_P2_frac_scan, P2_frac_scan, op=MPI.SUM, root=0)
comm.Reduce(local_P2_frac_scan_err, P2_frac_scan_err, op=MPI.SUM, root=0)
comm.Reduce(local_E_frac_scan, E_frac_scan, op=MPI.SUM, root=0)
comm.Reduce(local_E_frac_scan_err, E_frac_scan_err, op=MPI.SUM, root=0)
comm.Reduce(local_G_frac_scan, G_frac_scan, op=MPI.SUM, root=0)
comm.Reduce(local_G_frac_scan_err, G_frac_scan_err, op=MPI.SUM, root=0)
comm.Reduce(local_Q_scan, Q_scan, op=MPI.SUM, root=0)
comm.Reduce(local_Q_scan_err, Q_scan_err, op=MPI.SUM, root=0)

comm.Barrier()

if rank == 0:
    print("Gathered")
#%% Calculate and plot quantities vs time

if rank == 0:
    # Plot P2 fraction vs kapt
    plt.figure(figsize=(8,6))
    plt.errorbar(kapt_vals, P2_frac_scan, yerr=P2_frac_scan_err, marker='o', linestyle='-', markersize=10, label = '$P_{ZF}^2/P^2$',
                elinewidth=2, capthick=1, capsize=4)
    plt.fill_between(
        kapt_vals,
        P2_frac_scan - P2_frac_scan_err,
        P2_frac_scan + P2_frac_scan_err,
        color='grey',
        alpha=0.3,
        label='Error band'
    )
    plt.xlabel('$\\kappa_T$')
    plt.ylabel('$P_{ZF}^2/P^2$')
    plt.title('$P_{ZF}^2/P^2$ vs $\\kappa_T$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(datadir+'zonal_P2_frac_vs_kapt.pdf',dpi=100)
    plt.show()

    # Plot total energy fraction vs kapt
    plt.figure(figsize=(8,6))
    plt.errorbar(kapt_vals, E_frac_scan, yerr=E_frac_scan_err, marker='o', linestyle='-', markersize=10, label = '$\\mathcal{E}_{ZF}/\\mathcal{E}$',
                elinewidth=2, capthick=1, capsize=4)
    plt.fill_between(
        kapt_vals,
        E_frac_scan - E_frac_scan_err,
        E_frac_scan + E_frac_scan_err,
        color='grey',
        alpha=0.3,
        label='Error band'
    )
    plt.xlabel('$\\kappa_T$')
    plt.ylabel('$\\mathcal{E}_{ZF}/\\mathcal{E}$')
    plt.title('$\\mathcal{E}_{ZF}/\\mathcal{E}$ vs $\\kappa_T$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(datadir+'zonal_energy_frac_vs_kapt.pdf',dpi=100)
    plt.show()

    # Plot generalized energy fraction vs kapt
    plt.figure(figsize=(8,6))
    plt.errorbar(kapt_vals, G_frac_scan, yerr=G_frac_scan_err, marker='o', linestyle='-', markersize=10, label = '$\\mathcal{G}_{ZF}/\\mathcal{G}$',
                elinewidth=2, capthick=1, capsize=4)
    plt.fill_between(
        kapt_vals,
        G_frac_scan - G_frac_scan_err,
        G_frac_scan + G_frac_scan_err,
        color='grey',
        alpha=0.3,
        label='Error band'
    )
    plt.xlabel('$\\kappa_T$')
    plt.ylabel('$\\mathcal{G}_{ZF}/\\mathcal{G}$')
    plt.title('$\\mathcal{G}_{ZF}/\\mathcal{G}$ fraction vs $\\kappa_T$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(datadir+'zonal_generalized_energy_frac_vs_kapt.pdf',dpi=100)
    plt.show()

    # Plot Q vs kapt
    plt.figure(figsize=(8,6))
    plt.errorbar(kapt_vals, Q_scan, yerr=Q_scan_err, marker='o', linestyle='-', markersize=10, label = '$\\mathcal{Q}$',
                elinewidth=2, capthick=1, capsize=4)
    plt.fill_between(
        kapt_vals,
        Q_scan - Q_scan_err,
        Q_scan + Q_scan_err,
        color='grey',
        alpha=0.3,
        label='Error band'
    )
    plt.xlabel('$\\kappa_T$')
    plt.ylabel('$\\mathcal{Q}$')
    plt.title('$\\mathcal{Q}$ vs $\\kappa_T$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(datadir+'Q_vs_kapt.pdf',dpi=100)
    plt.show()
