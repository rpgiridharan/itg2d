#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist
from modules.mlsarray import irft2np as original_irft2np, rft2np as original_rft2np, irftnp as original_irftnp, rftnp as original_rftnp
from modules.gamma import gam_max
import os
from functools import partial
from mpi4py import MPI
import glob

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 3  

#%% Load the HDF5 file
comm.Barrier()
datadir='data_scan/'
kapt=0.85
D=0.1
pattern = datadir + f'out_kapt_{str(kapt).replace(".", "_")}_D_{str(D).replace(".", "_")}*.h5'
files = glob.glob(pattern)
if not files:
    print(f"No file found for kappa_T = {kapt}")
else:
    file_name = files[0]
it = -1

with h5.File(file_name, 'r', swmr=True) as fl:
    t = fl['fields/t'][:]
    nt= len(t)
    Omk = np.mean(fl['fields/Omk'][-(int(nt/2)):],axis=0)
    Pk = np.mean(fl['fields/Pk'][-(int(nt/2)):],axis=0)
    Ombar = np.mean(fl['zonal/Ombar'][-(int(nt/2)):],axis=0)
    Pbar = np.mean(fl['zonal/Pbar'][-(int(nt/2)):],axis=0)
    PiP = np.mean(fl['fluxes/RP'][-(int(nt/2)):],axis=0)
    Q = np.mean(fl['fluxes/Q'][-(int(nt/2)):],axis=0)
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
slky=np.s_[1:int(Ny/2)-1]
gammax=gam_max(kx,ky,kapn,kapt,kapb,D,HP,HPhi,slky)
t=t*gammax

nt = len(t) - (len(t) % size)
if rank == 0:
    print("nt: ", nt)

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

    nltermOmg=-kx**2*(rft2np(dxphi*dyP))+kx*ky*(rft2np(dxphi*dxP))-kx*ky*(rft2np(dyphi*dyP))+ky**2*(rft2np(dyphi*dxP))
    nltermP=rft2np(dxphi*dynOmg-dyphi*dxnOmg)

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

print(Omk.shape)

delk = ky[1]-ky[0]
kp = np.sqrt(np.abs(kx)**2 + np.abs(ky)**2)
k = np.linspace(np.min(kp), np.max(kp), num=int(np.max(kp)/delk))

Pik_phi = spectrum(Omk, Pk, kx, ky, k, delk, flag='Pik_phi')
Pik_d = spectrum(Omk, Pk, kx, ky, k, delk, flag='Pik_d')
Pik = Pik_phi + Pik_d

fk = spectrum(Omk, Pk, kx, ky, k, delk, flag='fk')
k_max_fk = k[np.argmax(fk)]

dk = spectrum(Omk, Pk, kx, ky, k, delk, flag='dk')

#%% Plots

plt.figure()
plt.plot(k[1:-1], Pik[1:-1], label = '$\\Pi_{k}$')
plt.plot(k[1:-1], Pik_phi[1:-1], label = '$\\Pi_{k,\\phi}$')
plt.plot(k[1:-1], Pik_d[1:-1], label = '$\\Pi_{k,d}$')
plt.axvline(x=k_max_fk, color='k', linestyle=':', linewidth=2)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel('$\\Pi_k$')
plt.title('$\\Pi_k$')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'energy_flux.png', dpi=600)
else:
    plt.savefig(datadir+"energy_flux_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

plt.figure()
plt.plot(k[1:-1], fk[1:-1], label = '$\\mathcal{f}_{k,total}$')
plt.axvline(x=k_max_fk, color='k', linestyle=':', linewidth=2, label=f'$k_{{max}}={k_max_fk:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel('$\\mathcal{f}_k$')
plt.title('$\\mathcal{f}_k$')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'energy_injection.png', dpi=600)
else:
    plt.savefig(datadir+"energy_injection_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

plt.figure()
plt.plot(k[1:-1], dk[1:-1], label = '$\\mathcal{d}_{k,total}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel('$\\mathcal{d}_k$')
plt.title('$\\mathcal{d}_k$')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'dissipation.png', dpi=600)
else:
    plt.savefig(datadir+"dissipation_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()