#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist, irft2, rft2, irft, rft
import os
from functools import partial
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 3  

#%% Load the HDF5 file
datadir = 'data/'
comm.Barrier()
file_name = datadir+'out_kapt_1_2_chi_0_1_D_0_0_e0_H_1_0_em3.h5'
it = -1
with h5.File(file_name, 'r', swmr=True) as fl:
    Omk = fl['fields/Omk'][0]
    Pk = fl['fields/Pk'][0]
    Ombar = fl['zonal/Ombar'][0]
    Pbar = fl['zonal/Pbar'][0]
    PiP = fl['fluxes/PiP'][0]
    Q = fl['fluxes/Q'][0]
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

nt = len(t) - (len(t) % size)
if rank == 0:
    print("nt: ", nt)

#%% Functions for energy, enstrophy and entropy

def K(Omk, kpsq):
    ''' Returns the total kinetic energy of the system'''
    E = np.sum(np.abs(-Omk)**2/kpsq).item()
    return np.real(E)

def K_ZF(Omk, kpsq, slbar):
    ''' Returns the zonal kinetic energy of the system'''
    E_ZF = np.sum(np.abs(-Omk[slbar])**2/kpsq[slbar]).item()
    return np.real(E_ZF)

def W(Omk):
    ''' Returns the total enstrophy of the system'''
    Wtemp = np.sum(np.abs(Omk)**2).item()
    return Wtemp
    
def W_ZF(Omk, slbar):
    ''' Returns the zonal enstrophy of the system'''
    W_ZFtemp = np.sum(np.abs(Omk[slbar])**2).item()
    return W_ZFtemp

def S(Omk, kpsq):
    ''' Returns the hyd. entropy of the system'''
    ek = np.abs(Omk)**2/kpsq
    pk = ek/np.sum(ek)
    pk = pk[pk > 0]
    S = -np.sum(pk*np.log2(pk)).item()
    return S

def sigma(Omk, kpsq):
    ''' Returns the entropy production rate of the system'''
    #-Q/T^2delT/delx
    ek = np.abs(Omk)**2/kpsq
    pk = ek/np.sum(ek)
    pk = pk[pk > 0]
    sig = -np.sum(pk*np.log2(pk)).item()
    return sig

#%% Calculate and plot quantities vs time

if rank == 0:
    P2_t = np.zeros(nt)
    P2_ZF_t = np.zeros(nt)
    energy_t = np.zeros(nt)
    energy_ZF_t = np.zeros(nt)
    enstrophy_t = np.zeros(nt)
    enstrophy_ZF_t = np.zeros(nt)
    entropy_t = np.zeros(nt)
    Ombar_t = np.zeros(nt)
    R_t = np.zeros(nt)
    PiP_t = np.zeros(nt)
    Q_t = np.zeros(nt)
    # Split range(nt) into 'size' sized chunks
    indices = np.array_split(range(nt), size) 
else:
    P2_t = None
    P2_ZF_t = None
    energy_t = None
    energy_ZF_t = None
    enstrophy_t = None
    enstrophy_ZF_t = None
    entropy_t = None
    Ombar_t = None
    R_t = None
    PiP_t = None
    Q_t = None
    indices = None

local_indices = comm.scatter(indices, root=0)

# Initialize local arrays for each process
P2_local = np.zeros(len(local_indices), dtype=np.float64)
P2_ZF_local = np.zeros(len(local_indices), dtype=np.float64)
energy_local = np.zeros(len(local_indices), dtype=np.float64)
energy_ZF_local = np.zeros(len(local_indices), dtype=np.float64)
enstrophy_local = np.zeros(len(local_indices), dtype=np.float64)
enstrophy_ZF_local = np.zeros(len(local_indices), dtype=np.float64)
entropy_local = np.zeros(len(local_indices), dtype=np.float64)
Ombar_local = np.zeros(len(local_indices), dtype=np.float64)
R_local = np.zeros(len(local_indices), dtype=np.float64)
PiP_local = np.zeros(len(local_indices), dtype=np.float64)
Q_local = np.zeros(len(local_indices), dtype=np.float64)

with h5.File(file_name, 'r', swmr=True) as fl:
    for idx, i in enumerate(local_indices):
        print(f"Rank {rank} processing time step {i}")
        Omk = fl['fields/Omk'][i]
        Pk = fl['fields/Pk'][i]
        Ombar = fl['zonal/Ombar'][i]
        Pbar = fl['zonal/Pbar'][i]
        R = fl['fluxes/R'][i]
        PiP = fl['fluxes/PiP'][i]
        Q = fl['fluxes/Q'][i]

        kpsq = kx**2 + ky**2

        # Calculate the consv quantities and fluxes
        P2_local[idx] = np.sum(np.abs(Pk)**2)
        P2_ZF_local[idx] = np.sum(np.abs(Pk[slbar])**2)
        energy_local[idx] = K(Omk, kpsq)
        energy_ZF_local[idx] = K_ZF(Omk, kpsq, slbar)
        enstrophy_local[idx] = W(Omk)
        enstrophy_ZF_local[idx] = W_ZF(Omk, slbar)
        entropy_local[idx] = S(Omk, kpsq)
        Ombar_local[idx] = np.mean(Ombar)
        R_local[idx] = np.mean(R)
        PiP_local[idx] = np.mean(PiP)
        Q_local[idx] = np.mean(Q)

# Gather results from all processes
comm.Gather(P2_local, P2_t, root=0)
comm.Gather(P2_ZF_local, P2_ZF_t, root=0)
comm.Gather(energy_local, energy_t, root=0)
comm.Gather(energy_ZF_local, energy_ZF_t, root=0)
comm.Gather(enstrophy_local, enstrophy_t, root=0)
comm.Gather(enstrophy_ZF_local, enstrophy_ZF_t, root=0)
comm.Gather(entropy_local, entropy_t, root=0)
comm.Gather(Ombar_local, Ombar_t, root=0)
comm.Gather(R_local, R_t, root=0)
comm.Gather(PiP_local, PiP_t, root=0)
comm.Gather(Q_local, Q_t, root=0)

comm.Barrier()

if rank == 0:
    print("Gathered")

if rank == 0:
    P2_turb_t = P2_t - P2_ZF_t
    energy_turb_t = energy_t - energy_ZF_t
    enstrophy_turb_t = enstrophy_t - enstrophy_ZF_t

    # Plot variance(P) vs time
    plt.figure(figsize=(8,6))
    plt.semilogy(t[:nt], P2_t, label = '$P_{total}$')
    plt.semilogy(t[:nt], P2_ZF_t, label = '$P_{ZF}^2$')
    plt.semilogy(t[:nt], P2_t, label = '$P_{turb}^2$')
    plt.xlabel('$t$')
    plt.ylabel('$P^2$')
    plt.title('$P^2$ vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'P2_vs_t.png',dpi=600)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'P2_vs_t_').replace('.h5', '.png'),dpi=600)
    plt.show()

    # Plot kinetic energy vs time
    plt.figure(figsize=(8,6))
    plt.semilogy(t[:nt], energy_t, label = '$\\mathcal{E}_{total}$')
    plt.semilogy(t[:nt], energy_ZF_t, label = '$\\mathcal{E}_{ZF}$')
    plt.semilogy(t[:nt], energy_turb_t, label = '$\\mathcal{E}_{turb}$')
    plt.xlabel('$t$')
    plt.ylabel('$\\mathcal{E}$')
    plt.title('Kinetic Energy vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'energy_vs_t.png',dpi=600)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'energy_vs_t_').replace('.h5', '.png'),dpi=600)
    plt.show()

    # Plot kinetic energy vs time
    plt.figure(figsize=(8,6))
    plt.semilogy(t[:nt], energy_ZF_t/energy_t, label = '$\\mathcal{E}_{ZF}/\\mathcal{E}$')
    plt.xlabel('$t$')
    plt.ylabel('$\\mathcal{E}_{ZF}/\\mathcal{E}$')
    plt.title('Zonal energy fraction vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    # if file_name.endswith('out.h5'):
    #     plt.savefig(datadir+'energy_vs_t.png',dpi=600)
    # else:
    #     plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'energy_vs_t_').replace('.h5', '.png'),dpi=600)
    plt.show()

    # Plot enstrophy vs time
    plt.figure(figsize=(8,6))
    plt.semilogy(t[:nt], enstrophy_t, label = '$\\mathcal{W}_{total}$')
    plt.semilogy(t[:nt], enstrophy_ZF_t, label = '$\\mathcal{W}_{ZF}$')
    plt.semilogy(t[:nt], enstrophy_turb_t, label = '$\\mathcal{W}_{turb}$')
    plt.xlabel('$t$')
    plt.ylabel('$\\mathcal{W}$')
    plt.title('Enstrophy vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'enstrophy_vs_t.png',dpi=600)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'enstrophy_vs_t_').replace('.h5', '.png'), dpi=600)
    plt.show()

    # Plot entropy vs time
    plt.figure(figsize=(8,6))
    plt.semilogy(t[:nt], entropy_t, label = '$\\mathcal{S}$')
    plt.xlabel('$t$')
    plt.ylabel('$\\mathcal{S}=-\\sum_{\\mathbf{k}}p_{\\mathbf{k}}\\log p_{\\mathbf{k}}$')
    plt.title('Entropy vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    # if file_name.endswith('out.h5'):
    #     plt.savefig(datadir+'entropy_vs_t.png',dpi=600)
    # else:
    #     plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'entropy_vs_t_').replace('.h5', '.png'), dpi=600)
    plt.show()

    # # Plot R vs time
    # plt.figure(figsize=(8,6))
    # plt.plot(t[:nt], R_t, '-', label = '$\\Pi_\\phi$')
    # plt.xlabel('$t$')
    # plt.ylabel('$\\Pi_\\phi$')
    # plt.title('$\\Pi_\\phi$ vs t')
    # plt.grid()
    # plt.legend()
    # plt.tight_layout()
    # if file_name.endswith('out.h5'):
    #     plt.savefig(datadir+'R_vs_t.png',dpi=600)
    # else:
    #     plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'R_vs_t_').replace('.h5', '.png'), dpi=600)
    # plt.show()

    # # Plot Pi vs time
    # plt.figure(figsize=(8,6))
    # plt.plot(t[:nt], PiP_t, '-', label = '$\\Pi_P$')
    # plt.xlabel('$t$')
    # plt.ylabel('$\\Pi_P$')
    # plt.title('$\\Pi_P$ vs t')
    # plt.grid()
    # plt.legend()
    # plt.tight_layout()
    # if file_name.endswith('out.h5'):
    #     plt.savefig(datadir+'PiP_vs_t.png',dpi=600)
    # else:
    #     plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'PiP_vs_t_').replace('.h5', '.png'), dpi=600)
    # plt.show()

    # Plot Q vs time
    plt.figure(figsize=(8,6))
    plt.semilogy(t[:nt], Q_t, '-', label = '$\\mathcal{Q}$')
    plt.xlabel('$t$')
    plt.ylabel('$\\mathcal{Q}$')
    plt.title('$\\mathcal{Q}$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'Q_vs_t.png',dpi=600)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'Q_vs_t_').replace('.h5', '.png'), dpi=600)
    plt.show()