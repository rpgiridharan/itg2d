#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist
from modules.mlsarray import irft2np as original_irft2np, rft2np as original_rft2np, irftnp as original_irftnp, rftnp as original_rftnp
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
file_name = datadir+'out_kapt_0_9_chi_0_1_H_1_0_em3.h5'
it = -1
with h5.File(file_name, 'r', swmr=True) as fl:
    Omk = fl['fields/Omk'][it]
    Pk = fl['fields/Pk'][it]
    Ombar = fl['zonal/Ombar'][it]
    Pbar = fl['zonal/Pbar'][it]
    PiP = fl['fluxes/PiP'][it]
    Q = fl['fluxes/Q'][it]
    t = fl['fields/t'][:]
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx = fl['params/Npx'][()]
    Npy = fl['params/Npy'][()]
    chi = fl['params/chi'][()]
    HP = fl['params/HP'][()]
    HPhi = fl['params/HPhi'][()]
    a = fl['params/a'][()]
    b = fl['params/b'][()]
    kapt = fl['params/kapt'][()]
    kapn = fl['params/kapn'][()]
    kapb = fl['params/kapb'][()]

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]

nt = len(t) - (len(t) % size)
if rank == 0:
    print("nt: ", nt)

#%% Functions for energy, enstrophy and entropy

irft2np = partial(original_irft2np,Npx=Npx,Npy=Npy,Nx=Nx,sl=sl)
rft2np = partial(original_rft2np,sl=sl)
irftnp = partial(original_irftnp,Npx=Npx,Nx=Nx)
rftnp = partial(original_rftnp,Nx=Nx)

def T1(Omk, Pk, kpsq):
    ''' Returns T1'''
    Phik = -Omk/kpsq
    dyphi=irft2np(1j*ky*Phik)
    dxphi=irft2np(1j*kx*Phik)
    dyP=irft2np(1j*ky*Pk)
    dxP=irft2np(1j*kx*Pk)
    term = np.sum(np.conj(1j*ky*Phik)*rft2np(dxphi*dyP-dyphi*dxP)).item()
    return np.real(term)

def T2(Omk, Pk, kpsq):
    ''' Returns T2'''
    Phik = -Omk/kpsq
    dyphi=irft2np(1j*ky*Phik)
    dxphi=irft2np(1j*kx*Phik)
    sigk=np.sign(ky)
    fac=sigk+kpsq
    dxnOmg=irft2np(1j*kx*fac*Phik)
    dynOmg=irft2np(1j*ky*fac*Phik)

    term = np.sum(1j*ky*rft2np(dxphi*dynOmg-dyphi*dxnOmg)*np.conj(Pk)/(1+kpsq)).item()
    return np.real(term)

def T3(Omk, Pk, kpsq, kx, ky):
    ''' Returns T3'''
    Phik = -Omk/kpsq
    dyphi=irft2np(1j*ky*Phik)
    dxphi=irft2np(1j*kx*Phik)
    dyP=irft2np(1j*ky*Pk)
    dxP=irft2np(1j*kx*Pk)

    nlterm=-kx**2*(rft2np(dxphi*dyP))+kx*ky*(rft2np(dxphi*dxP))-kx*ky*(rft2np(dyphi*dyP))+ky**2*(rft2np(dyphi*dxP))
    term = np.sum(1j*ky*nlterm*np.conj(Pk)/(1+kpsq)).item()
    return np.real(term)

def L(Omk, Pk, kpsq):
    ''' Returns L'''
    Phik = -Omk/kpsq
    term1 = np.sum((kapn+kapt)*ky**2*np.abs(Phik)**2 - 2*(5/3)*kapb*ky**2*np.conj(Pk)*Phik).item()
    term2 = np.sum((kapb-kapn)*ky**2*Phik*np.conj(Pk)+(kapn+kapt)*ky**2*kpsq*Phik*np.conj(Pk)+kapb*ky**2*np.abs(Pk)**2/(1+kpsq)).item()
    return np.real(term1 + term2)

def Dchi(Omk, Pk, kpsq, ky):
    ''' Returns D'''
    Phik = -Omk/kpsq
    term1 = chi*np.sum(kpsq*1j*ky*Phik*np.conj(Pk)).item()
    term2 = chi*np.sum(1j*ky*kpsq**2*(a*Phik-b*Pk)*np.conj(Pk)/(1+kpsq)).item()
    return np.real(term1+term2)

def DH(Omk, Pk, kpsq, ky):
    ''' Returns D'''
    Phik = -Omk/kpsq
    sigk = np.sign(ky)
    term = (HP+HPhi)*np.sum(sigk*1j*ky*kpsq**(-3)*Phik*np.conj(Pk)).item()
    return np.real(term)

#%% Calculate and plot quantities vs time

if rank == 0:
    T1_t = np.zeros(nt)
    T2_t = np.zeros(nt)
    T3_t = np.zeros(nt)
    L_t = np.zeros(nt)
    DH_t = np.zeros(nt)
    Dchi_t = np.zeros(nt)
    R_t = np.zeros(nt)
    PiP_t = np.zeros(nt)
    Q_t = np.zeros(nt)
    # Split range(nt) into 'size' sized chunks
    indices = np.array_split(range(nt), size) 
else:
    T1_t = None
    T2_t = None
    T3_t = None
    L_t = None
    DH_t = None
    Dchi_t = None
    R_t = None
    PiP_t = None
    Q_t = None
    indices = None

local_indices = comm.scatter(indices, root=0)

# Initialize local arrays for each process
T1_local = np.zeros(len(local_indices), dtype=np.float64)
T2_local = np.zeros(len(local_indices), dtype=np.float64)
T3_local = np.zeros(len(local_indices), dtype=np.float64)
L_local = np.zeros(len(local_indices), dtype=np.float64)
DH_local = np.zeros(len(local_indices), dtype=np.float64)
Dchi_local = np.zeros(len(local_indices), dtype=np.float64)
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
        T1_local[idx] = T1(Omk, Pk, kpsq)
        T2_local[idx] = T2(Omk, Pk, kpsq)
        T3_local[idx] = T3(Omk, Pk, kpsq, kx, ky)
        L_local[idx] = L(Omk, Pk, kpsq)
        DH_local[idx] = DH(Omk, Pk, kpsq, ky)
        Dchi_local[idx] = Dchi(Omk, Pk, kpsq, ky)
        R_local[idx] = np.mean(R)
        PiP_local[idx] = np.mean(PiP)
        Q_local[idx] = np.mean(Q)

# Gather results from all processes
comm.Gather(T1_local, T1_t, root=0)
comm.Gather(T2_local, T2_t, root=0)
comm.Gather(T3_local, T3_t, root=0)
comm.Gather(L_local, L_t, root=0)
comm.Gather(DH_local, DH_t, root=0)
comm.Gather(Dchi_local, Dchi_t, root=0)
comm.Gather(R_local, R_t, root=0)
comm.Gather(PiP_local, PiP_t, root=0)
comm.Gather(Q_local, Q_t, root=0)

comm.Barrier()

if rank == 0:
    print("Gathered")

if rank == 0:

    # Plot T1 vs time
    plt.figure(figsize=(8,6))
    plt.plot(t[:nt], T1_t, label = '$T_1$')
    plt.xlabel('$t$')
    plt.ylabel('$T_1$')
    plt.title('$T_1$ vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'T1_vs_t.pdf',dpi=100)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'T1_vs_t_').replace('.h5', '.pdf'),dpi=100)
    plt.show()

    # Plot T2 vs time
    plt.figure(figsize=(8,6))
    plt.plot(t[:nt], T2_t, label = '$T_2$')
    plt.xlabel('$t$')
    plt.ylabel('$T_2$')
    plt.title('$T_2$ vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'T2_vs_t.pdf',dpi=100)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'T2_vs_t_').replace('.h5', '.pdf'),dpi=100)
    plt.show()

    # Plot T3 vs time
    plt.figure(figsize=(8,6))
    plt.plot(t[:nt], T3_t, label = '$T_3$')
    plt.xlabel('$t$')
    plt.ylabel('$T_3$')
    plt.title('$T_3$ vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'T3_vs_t.pdf',dpi=100)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'T3_vs_t_').replace('.h5', '.pdf'),dpi=100)
    plt.show()

    # Plot T1+T2 vs time
    plt.figure(figsize=(8,6))
    plt.plot(t[:nt], T1_t + T2_t, label = '$T_1 + T_2$')
    plt.xlabel('$t$')
    plt.ylabel('$T_1 + T_2$')
    plt.title('$T_1 + T_2$ vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'T1_plus_T2_vs_t.pdf',dpi=100)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'T1_plus_T2_vs_t_').replace('.h5', '.pdf'),dpi=100)
    plt.show()

    # Plot T1+T2+T3 vs time
    plt.figure(figsize=(8,6))
    plt.plot(t[:nt], T1_t + T2_t + T3_t, label = '$T_1 + T_2 + T_3$')
    plt.xlabel('$t$')
    plt.ylabel('$T_1 + T_2 + T_3$')
    plt.title('$T_1 + T_2 + T_3$ vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'T1_plus_T2_plus_T3_vs_t.pdf',dpi=100)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'T1_plus_T2_plus_T3_vs_t_').replace('.h5', '.pdf'),dpi=100)
    plt.show()

    # Plot L vs time
    plt.figure(figsize=(8,6))
    plt.plot(t[:nt], L_t, label = '$L$')
    plt.xlabel('$t$')
    plt.ylabel('$L$')
    plt.title('$L$ vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'L_vs_t.pdf',dpi=100)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'L_vs_t_').replace('.h5', '.pdf'),dpi=100)
    plt.show()

    # Plot Dchi vs time
    plt.figure(figsize=(8,6))
    plt.plot(t[:nt], Dchi_t, label = '$D_\\chi$')
    plt.xlabel('$t$')
    plt.ylabel('$D_\\chi$')
    plt.title('$D_\\chi$ vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'Dchi_vs_t.pdf',dpi=100)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'Dchi_vs_t_').replace('.h5', '.pdf'),dpi=100)
    plt.show()

    # Plot DH vs time
    plt.figure(figsize=(8,6))
    plt.plot(t[:nt], DH_t, label = '$D_H$')
    plt.xlabel('$t$')
    plt.ylabel('$D_H$')
    plt.title('$D_H$ vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'DH_vs_t.pdf',dpi=100)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'DH_vs_t_').replace('.h5', '.pdf'),dpi=100)
    plt.show()

    # Plot Dchi + DH vs time
    plt.figure(figsize=(8,6))
    plt.plot(t[:nt], Dchi_t + DH_t, label = '$D_\\chi + D_H$')
    plt.xlabel('$t$')
    plt.ylabel('$D_\\chi + D_H$')
    plt.title('$D_\\chi + D_H$ vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'Dchi_plus_DH_vs_t.pdf',dpi=100)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'Dchi_plus_DH_vs_t_').replace('.h5', '.pdf'),dpi=100)
    plt.show()

    # Plot T1 + T2 + T3 + L + Dchi + DH vs time
    plt.figure(figsize=(8,6))
    plt.plot(t[:nt], T1_t + T2_t + T3_t + L_t + Dchi_t + DH_t, label = '$T+L+D$')
    plt.xlabel('$t$')
    plt.ylabel('$T+L+D$')
    plt.title('$T+L+D$ vs t')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'T_plus_L_plus_D_vs_t.pdf',dpi=100)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'T_plus_L_plus_D_vs_t_').replace('.h5', '.pdf'),dpi=100)
    plt.show()

    # Plot dQdt vs time
    dQdt = np.gradient(Q_t[:nt], t[:nt])
    plt.figure(figsize=(8,6))
    plt.plot(t[:nt], dQdt, '-', label = '$\\frac{d\\mathcal{Q}}{dt}$')
    plt.xlabel('$t$')
    plt.ylabel('$\\frac{d\\mathcal{Q}}{dt}$')
    plt.title('$\\frac{d\\mathcal{Q}}{dt}$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'dQdt_vs_t.pdf',dpi=100)
    else:
        plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'dQdt_vs_t_').replace('.h5', '.pdf'), dpi=100)
    plt.show()

    # Plot Q vs time
    plt.figure(figsize=(8,6))
    plt.semilogy(t[:nt], Q_t, '-', label = '$\\mathcal{Q}$')
    plt.xlabel('$t$')
    plt.ylabel('$\\mathcal{Q}$')
    plt.title('$\\mathcal{Q}$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    # if file_name.endswith('out.h5'):
        # plt.savefig(datadir+'Q_vs_t.pdf',dpi=100)
    # else:
        # plt.savefig(datadir+file_name.split('/')[-1].replace('out_', 'Q_vs_t_').replace('.h5', '.pdf'), dpi=100)
    plt.show()