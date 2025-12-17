#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist
from modules.mlsarray import irft2np as original_irft2np, rft2np as original_rft2np, irftnp as original_irftnp, rftnp as original_rftnp
from modules.gamma import gam_max, ky_max
import os
from functools import partial
from mpi4py import MPI
import glob
from scipy.stats import gaussian_kde

# Initialize MPI
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
    'legend.fontsize': 16
})

#%% Load the HDF5 file

datadir='data/'
# kapt=1.0
# D=0.1
# Np=1024
# pattern = datadir + f'out_kapt_{str(kapt).replace(".", "_")}_D_{str(D).replace(".", "_")}*_{Np}x{Np}.h5'
# files = glob.glob(pattern)
# if not files:
#     print(f"No file found for kappa_T = {kapt}")
# else:
#     file_name = files[0]

file_name = datadir + 'out_kapt_2_0_D_0_02_H_1_0_em5_1024x1024.h5'

with h5.File(file_name, 'r', swmr=True) as fl:
    t = fl['fields/t'][:]
    nt= len(t)
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx = fl['params/Npx'][()]
    Npy = fl['params/Npy'][()]
    kapn = fl['params/kapn'][()]
    kapt = fl['params/kapt'][()]
    kapb = fl['params/kapb'][()]
    D = fl['params/D'][()]
    HP = fl['params/HP'][()]
    HPhi = fl['params/HPhi'][()]

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
gammax=gam_max(kx,ky,kapn,kapt,kapb,D,HP,HPhi)
t=t*gammax
k_lin = ky_max(kx,ky,kapn,kapt,kapb,D,HP,HPhi)

#%% Functions for energy, enstrophy and entropy

irft2np = partial(original_irft2np,Npx=Npx,Npy=Npy,Nx=Nx,sl=sl)
rft2np = partial(original_rft2np,sl=sl)
irftnp = partial(original_irftnp,Npx=Npx,Nx=Nx)
rftnp = partial(original_rftnp,Nx=Nx)

def spectrum(Omk, Pk, kx, ky, k, delk, flag='Pik'):
    ''' Returns the RHS of the model equations'''
    kpsq = kx**2 + ky**2
    Phik = -Omk/kpsq
    dyphi=irft2np(1j*ky*Phik)
    dxphi=irft2np(1j*kx*Phik)
    dyP=irft2np(1j*ky*Pk)
    dxP=irft2np(1j*kx*Pk)
    sigk=np.sign(ky)
    fac=sigk+kpsq
    dxnOmg=irft2np(1j*kx*fac*Phik)
    dynOmg=irft2np(1j*ky*fac*Phik)

    nltermOmg=rft2np(dxphi*dynOmg-dyphi*dxnOmg)
    nltermP=-kx**2*(rft2np(dxphi*dyP))+kx*ky*(rft2np(dxphi*dxP))-kx*ky*(rft2np(dyphi*dyP))+ky**2*(rft2np(dyphi*dxP))

    ak = np.zeros_like(Omk)
    Ak = np.zeros(len(k))
    if flag=='Pik_phi':
        ak = np.real(np.conj(Phik)*nltermOmg)
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(kp<=k[i])])
    elif flag=='Pik_d':
        ak = np.real(np.conj(Phik)*nltermP)
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(kp<=k[i])])
    elif flag=='fk':
        ak = np.real(np.conj(Phik)*(-kapn*1j*ky*Phik+(kapn+kapt)*1j*ky*kpsq*Phik+kapb*1j*ky*Pk))
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(np.logical_and(kp>k[i], kp<=k[i]+delk))])/delk
    elif flag=='dk':
        ak = np.real(np.conj(Phik)*(D*kpsq*Phik+HPhi*(1+kpsq)/kpsq**2*Phik))
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(np.logical_and(kp>k[i], kp<=k[i]+delk))])/delk
    return Ak
    
#%% Calculate quantities

delk = ky[1]-ky[0]
kp = np.sqrt(np.abs(kx)**2 + np.abs(ky)**2)
k = np.linspace(np.min(kp), np.max(kp), num=int(np.max(kp)/delk))

# MPI parallelization

nt2 = int(nt/2)
nt2 = nt2 - (nt2 % size)
if rank == 0:
    indices = np.array_split(range(nt2), size)
else:
    indices = None

local_indices = comm.scatter(indices, root=0)
count_local = len(local_indices)

Pik_phi_t_local = np.zeros((count_local, len(k)))
Pik_d_t_local = np.zeros((count_local, len(k)))
fk_t_local = np.zeros((count_local, len(k)))
dk_t_local = np.zeros((count_local, len(k)))

with h5.File(file_name, 'r', swmr=True) as fl:
    for idx, it in enumerate(local_indices):
        print(f"Rank {rank} processing time step {it}")
        Omk = fl['fields/Omk'][it+nt//2]
        Pk = fl['fields/Pk'][it+nt//2]
        Pik_phi_t_local[idx,:] = spectrum(Omk, Pk, kx, ky, k, delk, flag='Pik_phi')
        Pik_d_t_local[idx,:] = spectrum(Omk, Pk, kx, ky, k, delk, flag='Pik_d')
        fk_t_local[idx,:] = spectrum(Omk, Pk, kx, ky, k, delk, flag='fk')
        dk_t_local[idx,:] = spectrum(Omk, Pk, kx, ky, k, delk, flag='dk')

# Gather results at root
Pik_phi_t = None
Pik_d_t = None
fk_t = None
dk_t = None
if rank == 0:
    Pik_phi_t = np.zeros((nt2, len(k)))
    Pik_d_t = np.zeros((nt2, len(k)))
    fk_t = np.zeros((nt2, len(k)))
    dk_t = np.zeros((nt2, len(k)))

comm.Gather(Pik_phi_t_local, Pik_phi_t, root=0)
comm.Gather(Pik_d_t_local, Pik_d_t, root=0)
comm.Gather(fk_t_local, fk_t, root=0)
comm.Gather(dk_t_local, dk_t, root=0)

if rank == 0:
    Pik_phi = np.mean(Pik_phi_t, axis=0)
    Pik_d = np.mean(Pik_d_t, axis=0)
    fk = np.mean(fk_t, axis=0)
    dk = np.mean(dk_t, axis=0)
    Pik = Pik_phi + Pik_d
    idx_k = np.argmax(fk)
    k_f = k[idx_k]

    # PDF of fluxes at k_f
    Pik_phi_series = Pik_phi_t[:, idx_k]
    Pik_d_series = Pik_d_t[:, idx_k]
    Pik_series = Pik_phi_series + Pik_d_series

    Pik_phi_series_norm = (Pik_phi_series - np.mean(Pik_phi_series)) / np.std(Pik_phi_series)
    Pik_d_series_norm = (Pik_d_series - np.mean(Pik_d_series)) / np.std(Pik_d_series)
    Pik_series_norm = (Pik_series - np.mean(Pik_series)) / np.std(Pik_series)

    # PDF of fluxes at k=kymax
    idx_k = np.argmin(np.abs(k - k_lin))
    Pik_phi_series_max = Pik_phi_t[:, idx_k]
    Pik_d_series_max = Pik_d_t[:, idx_k]
    Pik_series_max = Pik_phi_series_max + Pik_d_series_max

    Pik_phi_series_max_norm = (Pik_phi_series_max - np.mean(Pik_phi_series_max)) / np.std(Pik_phi_series_max)
    Pik_d_series_max_norm = (Pik_d_series_max - np.mean(Pik_d_series_max)) / np.std(Pik_d_series_max)
    Pik_series_max_norm = (Pik_series_max - np.mean(Pik_series_max)) / np.std(Pik_series_max)

#%% Plots
if rank == 0:
    plt.figure()
    plt.plot(k[1:-1], Pik[1:-1], label = '$\\Pi_{k}$')
    plt.plot(k[1:-1], Pik_phi[1:-1], label = '$\\Pi_{k,\\phi}$')
    plt.plot(k[1:-1], Pik_d[1:-1], label = '$\\Pi_{k,d}$')
    plt.axhline(0,color='k', linestyle='-', linewidth=1)
    plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
    plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2, label=f'$k_{{lin}}={k_lin:.2f}$')
    plt.xscale('log')
    plt.xlabel('$k$')
    plt.ylabel('$\\Pi_k$')
    plt.title(f'$\\Pi_k$ for $\\kappa_T={kapt:.2f}$')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'energy_flux.pdf', dpi=100)
    else:
        plt.savefig(datadir+"energy_flux_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
    plt.show()

    plt.figure()
    plt.plot(k[1:-1], fk[1:-1], label = '$\\mathcal{f}_{k,total}$')
    plt.axhline(0,color='k', linestyle='-', linewidth=1)
    plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
    plt.xscale('log')
    plt.xlabel('$k$')
    plt.ylabel('$\\mathcal{f}_k$')
    plt.title(f'$\\mathcal{{f}}_k$ for $\\kappa_T={kapt:.2f}$')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'energy_injection.pdf', dpi=100)
    else:
        plt.savefig(datadir+"energy_injection_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
    plt.show()

    plt.figure()
    plt.plot(k[1:-1], dk[1:-1], label = '$\\mathcal{d}_{k,total}$')
    plt.axhline(0,color='k', linestyle='-', linewidth=1)
    plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
    plt.xscale('log')
    plt.xlabel('$k$')
    plt.ylabel('$\\mathcal{d}_k$')
    plt.title(f'$\\mathcal{{d}}_k$ for $\\kappa_T={kapt:.2f}$')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'dissipation.pdf', dpi=100)
    else:
        plt.savefig(datadir+"dissipation_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
    plt.show()

    plt.figure()
    for series, label in zip([Pik_series_norm, Pik_phi_series_norm, Pik_d_series_norm], 
                            [r'$\Pi_{k}$', r'$\Pi_{k,\phi}$', r'$\Pi_{k,d}$']):
        kde = gaussian_kde(series)
        x_range = np.linspace(series.min(), series.max(), 200)
        plt.hist(series, bins=50, density=True, alpha=0.3, color='gray')
        plt.plot(x_range, kde(x_range), label=label)
    plt.xlabel('$\\frac{\\left(\\Pi_k-<\\Pi_k>\\right)}{\\sigma}$')
    plt.ylabel('PDF')
    plt.title(f'PDF of $\\Pi_k$ at $k_f={k_f:.2f}$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'energy_flux_pdf_.pdf', dpi=100)
    else:
        plt.savefig(datadir+"energy_flux_pdf_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
    plt.show()

    plt.figure()
    for series, label in zip([Pik_series_max_norm, Pik_phi_series_max_norm, Pik_d_series_max_norm], 
                            [r'$\Pi_{k}$', r'$\Pi_{k,\phi}$', r'$\Pi_{k,d}$']):
        kde = gaussian_kde(series)
        x_range = np.linspace(series.min(), series.max(), 200)
        plt.hist(series, bins=50, density=True, alpha=0.3, color='gray')
        plt.plot(x_range, kde(x_range), label=label)
    plt.xlabel('$\\frac{\\left(\\Pi_k-<\\Pi_k>\\right)}{\\sigma}$')
    plt.ylabel('PDF')
    plt.title(f'PDF of $\\Pi_k$ at $k_{{lin}}={k_lin:.2f}$')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_name.endswith('out.h5'):
        plt.savefig(datadir+'energy_flux_kymax_pdf_.pdf', dpi=100)
    else:
        plt.savefig(datadir+"energy_flux_kymax_pdf_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
    plt.show()

