#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
from modules.mlsarray import Slicelist, irft2np
from functools import partial
from modules.gamma_2d3c import gam_max
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#%% Load the HDF5 file

Npx=1024
datadir=f'data_2d3c/{Npx}/'
fname = datadir+'out_2d3c_kapt_2_0_D_0_1_kz_0_1.h5'

with h5.File(fname, 'r', swmr=True) as fl:
    Omk = fl['fields/Omk'][0]
    Pk = fl['fields/Pk'][0]
    Vk = fl['fields/Vk'][0]
    t = fl['fields/t'][:]
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx = fl['params/Npx'][()]
    Npy = fl['params/Npy'][()]
    kapn = fl['params/kapn'][()]
    kapt = fl['params/kapt'][()]
    kapb = fl['params/kapb'][()]
    if 'D' in fl['params']:
        D = fl['params/D'][()]
    elif 'chi' in fl['params']:
        chi = fl['params/chi'][()]
        D = chi
    kz = fl['params/kz'][()]

Nx, Ny = 2*Npx//3, 2*Npy//3
sl = Slicelist(Nx, Ny)
slbar = np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
gammax = gam_max(kx, ky, kapn, kapt, kapb, D, kz)
t = t * gammax
kpsq = kx**2 + ky**2
irft2 = partial(irft2np, Npx=int(Npx), Npy=int(Npy), Nx=int(Nx), sl=sl)

#%% Functions for energy, enstrophy and entropy

def E(Omk, ky, kpsq):
    ''' Returns the total energy of the system'''
    sigk = np.sign(ky)
    fac = sigk + kpsq
    Etemp = np.sum(fac*np.abs(Omk)**2/kpsq**2).item()
    return np.real(Etemp)

def E_ZF(Omk, ky, kpsq, slbar):
    ''' Returns the zonal energy of the system'''
    sigk = np.sign(ky)
    fac = sigk + kpsq
    E_ZFtemp = np.sum(fac[slbar]*np.abs(Omk[slbar])**2/kpsq[slbar]**2).item()
    return np.real(E_ZFtemp)

def K(Omk, kpsq):
    ''' Returns the kinetic energy of the system'''
    Ktemp = np.sum(np.abs(Omk)**2/kpsq).item()
    return np.real(Ktemp)

def K_ZF(Omk, kpsq, slbar):
    ''' Returns the zonal kinetic energy of the system'''
    K_ZFtemp = np.sum(np.abs(Omk[slbar])**2/kpsq[slbar]).item()
    return np.real(K_ZFtemp)

def W(Omk):
    ''' Returns the total enstrophy of the system'''
    Wtemp = np.sum(np.abs(Omk)**2).item()
    return Wtemp

def W_ZF(Omk, slbar):
    ''' Returns the zonal enstrophy of the system'''
    W_ZFtemp = np.sum(np.abs(Omk[slbar])**2).item()
    return W_ZFtemp

def G(Omk, Pk, ky, kpsq):
    ''' Returns the generalized energy of the system'''
    sigk = np.sign(ky)
    Phik = Omk/kpsq
    Gtemp = np.sum(np.abs(sigk*Phik+Pk)**2+kpsq*np.abs(Phik+Pk)**2).item()
    return np.real(Gtemp)

def G_ZF(Omk, Pk, ky, kpsq, slbar):
    ''' Returns the zonal generalized energy of the system'''
    sigk = np.sign(ky)
    Phik = Omk/kpsq
    G_ZFtemp = np.sum(np.abs(sigk[slbar]*Phik[slbar]+Pk[slbar])**2+kpsq[slbar]*np.abs(Phik[slbar]+Pk[slbar])**2).item()
    return np.real(G_ZFtemp)

def S(Omk, kpsq):
    ''' Returns the hyd. entropy of the system'''
    ek = np.abs(Omk)**2/kpsq
    pk = ek/np.sum(ek)
    pk = pk[pk > 0]
    S = -np.sum(pk*np.log2(pk)).item()
    return S

#%% Calculate quantities vs time using MPI

nt = len(t) - (len(t) % size)
if rank == 0:
    print("nt: ", nt)
    indices = np.array_split(range(nt), size)
else:
    indices = None

local_indices = comm.scatter(indices, root=0)

# Initialize local arrays for each process
P2_local = np.zeros(len(local_indices), dtype=np.float64)
P2_ZF_local = np.zeros(len(local_indices), dtype=np.float64)
V2_local = np.zeros(len(local_indices), dtype=np.float64)
V2_ZF_local = np.zeros(len(local_indices), dtype=np.float64)
energy_local = np.zeros(len(local_indices), dtype=np.float64)
energy_ZF_local = np.zeros(len(local_indices), dtype=np.float64)
kin_energy_local = np.zeros(len(local_indices), dtype=np.float64)
kin_energy_ZF_local = np.zeros(len(local_indices), dtype=np.float64)
enstrophy_local = np.zeros(len(local_indices), dtype=np.float64)
enstrophy_ZF_local = np.zeros(len(local_indices), dtype=np.float64)
gen_energy_local = np.zeros(len(local_indices), dtype=np.float64)
gen_energy_ZF_local = np.zeros(len(local_indices), dtype=np.float64)
entropy_local = np.zeros(len(local_indices), dtype=np.float64)
Q_local = np.zeros(len(local_indices), dtype=np.float64)
Ombar_local = np.zeros(len(local_indices), dtype=np.float64)
electric_reynolds_power_local = np.zeros(len(local_indices), dtype=np.float64)
diamagnetic_reynolds_power_local = np.zeros(len(local_indices), dtype=np.float64)
reynolds_power_local = np.zeros(len(local_indices), dtype=np.float64)

with h5.File(fname, 'r', swmr=True) as fl:
    for idx, i in enumerate(local_indices):
        print(f"Rank {rank} processing time step {i}")
        Omk = fl['fields/Omk'][i]
        Pk = fl['fields/Pk'][i]
        Vk = fl['fields/Vk'][i]

        Phik = -Omk / kpsq
        Om   = irft2(Omk)
        P    = irft2(Pk)
        vx   = irft2(-1j*ky*Phik)
        vy   = irft2(1j*kx*Phik)
        wx   = irft2(-1j*ky*Pk)
        Ombar = np.mean(Om, axis=1)
        Q     = np.mean(P*vx, axis=1)
        RPhi  = np.mean(vy*vx, axis=1)
        RP    = np.mean(vy*wx, axis=1)

        P2_local[idx] = np.sum(np.abs(Pk)**2)
        P2_ZF_local[idx] = np.sum(np.abs(Pk[slbar])**2)
        V2_local[idx] = np.sum(np.abs(Vk)**2)
        V2_ZF_local[idx] = np.sum(np.abs(Vk[slbar])**2)
        energy_local[idx] = E(Omk, ky, kpsq)
        energy_ZF_local[idx] = E_ZF(Omk, ky, kpsq, slbar)
        kin_energy_local[idx] = K(Omk, kpsq)
        kin_energy_ZF_local[idx] = K_ZF(Omk, kpsq, slbar)
        enstrophy_local[idx] = W(Omk)
        enstrophy_ZF_local[idx] = W_ZF(Omk, slbar)
        gen_energy_local[idx] = G(Omk, Pk, ky, kpsq)
        gen_energy_ZF_local[idx] = G_ZF(Omk, Pk, ky, kpsq, slbar)
        entropy_local[idx] = S(Omk, kpsq)
        Q_local[idx] = np.mean(Q)
        Ombar_local[idx] = np.mean(Ombar)
        electric_reynolds_power_local[idx] = np.mean(RPhi * Ombar)
        diamagnetic_reynolds_power_local[idx] = np.mean(RP * Ombar)
        reynolds_power_local[idx] = np.mean((RPhi + RP) * Ombar)

# Gather results from all processes
P2_t = P2_ZF_t = V2_t = V2_ZF_t = energy_t = energy_ZF_t = kin_energy_t = kin_energy_ZF_t = enstrophy_t = enstrophy_ZF_t = gen_energy_t = gen_energy_ZF_t = entropy_t = Q_t = Ombar_t = electric_reynolds_power_t = diamagnetic_reynolds_power_t = reynolds_power_t = None

if rank == 0:
    P2_t = np.zeros(nt)
    P2_ZF_t = np.zeros(nt)
    V2_t = np.zeros(nt)
    V2_ZF_t = np.zeros(nt)
    energy_t = np.zeros(nt)
    energy_ZF_t = np.zeros(nt)
    kin_energy_t = np.zeros(nt)
    kin_energy_ZF_t = np.zeros(nt)
    enstrophy_t = np.zeros(nt)
    enstrophy_ZF_t = np.zeros(nt)
    gen_energy_t = np.zeros(nt)
    gen_energy_ZF_t = np.zeros(nt)
    entropy_t = np.zeros(nt)
    Q_t = np.zeros(nt)
    Ombar_t = np.zeros(nt)
    electric_reynolds_power_t = np.zeros(nt)
    diamagnetic_reynolds_power_t = np.zeros(nt)
    reynolds_power_t = np.zeros(nt)

comm.Gather(P2_local, P2_t, root=0)
comm.Gather(P2_ZF_local, P2_ZF_t, root=0)
comm.Gather(V2_local, V2_t, root=0)
comm.Gather(V2_ZF_local, V2_ZF_t, root=0)
comm.Gather(energy_local, energy_t, root=0)
comm.Gather(energy_ZF_local, energy_ZF_t, root=0)
comm.Gather(kin_energy_local, kin_energy_t, root=0)
comm.Gather(kin_energy_ZF_local, kin_energy_ZF_t, root=0)
comm.Gather(enstrophy_local, enstrophy_t, root=0)
comm.Gather(enstrophy_ZF_local, enstrophy_ZF_t, root=0)
comm.Gather(gen_energy_local, gen_energy_t, root=0)
comm.Gather(gen_energy_ZF_local, gen_energy_ZF_t, root=0)
comm.Gather(entropy_local, entropy_t, root=0)
comm.Gather(Q_local, Q_t, root=0)
comm.Gather(Ombar_local, Ombar_t, root=0)
comm.Gather(electric_reynolds_power_local, electric_reynolds_power_t, root=0)
comm.Gather(diamagnetic_reynolds_power_local, diamagnetic_reynolds_power_t, root=0)
comm.Gather(reynolds_power_local, reynolds_power_t, root=0)

#%% Save computed results to HDF5 file (rank 0 only)

if rank == 0:
    print("Gathered")
    out_fname = datadir + fname.split('/')[-1].replace('out_', 'evol_')
    with h5.File(out_fname, 'w') as fl:
        fl.create_dataset('t', data=t[:nt])
        fl.create_dataset('P2_t', data=P2_t)
        fl.create_dataset('P2_ZF_t', data=P2_ZF_t)
        fl.create_dataset('V2_t', data=V2_t)
        fl.create_dataset('V2_ZF_t', data=V2_ZF_t)
        fl.create_dataset('energy_t', data=energy_t)
        fl.create_dataset('energy_ZF_t', data=energy_ZF_t)
        fl.create_dataset('kin_energy_t', data=kin_energy_t)
        fl.create_dataset('kin_energy_ZF_t', data=kin_energy_ZF_t)
        fl.create_dataset('enstrophy_t', data=enstrophy_t)
        fl.create_dataset('enstrophy_ZF_t', data=enstrophy_ZF_t)
        fl.create_dataset('gen_energy_t', data=gen_energy_t)
        fl.create_dataset('gen_energy_ZF_t', data=gen_energy_ZF_t)
        fl.create_dataset('entropy_t', data=entropy_t)
        fl.create_dataset('Q_t', data=Q_t)
        fl.create_dataset('Ombar_t', data=Ombar_t)
        fl.create_dataset('electric_reynolds_power_t', data=electric_reynolds_power_t)
        fl.create_dataset('diamagnetic_reynolds_power_t', data=diamagnetic_reynolds_power_t)
        fl.create_dataset('reynolds_power_t', data=reynolds_power_t)
        # Store metadata
        fl.attrs['fname'] = fname
        fl.attrs['datadir'] = datadir
    print(f"Saved computed results to {out_fname}")
