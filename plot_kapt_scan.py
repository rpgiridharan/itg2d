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

datadir = 'data_scan/'
kapt_vals=np.arange(0.2,0.85,0.05) 

P2_frac_scan = np.zeros(kapt_vals.shape)
P2_frac_scan_err = np.zeros(kapt_vals.shape)
E_frac_scan = np.zeros(kapt_vals.shape)
E_frac_scan_err = np.zeros(kapt_vals.shape)
G_frac_scan = np.zeros(kapt_vals.shape)
G_frac_scan_err = np.zeros(kapt_vals.shape)
Q_scan = np.zeros(kapt_vals.shape)
Q_scan_err = np.zeros(kapt_vals.shape)

for i,kapt in enumerate(kapt_vals):
    kapt = round(kapt, 3)  
    D = 0.1
    print(f'Processing kappa_T = {kapt}')
    pattern = datadir + f'out_kapt_{str(kapt).replace(".", "_")}_D_{str(D).replace(".", "_")}*.h5'
    files = glob.glob(pattern)
    if not files:
        print(f"No file found for kappa_T = {kapt}")
        continue
    else:
        file_name = files[0]

    with h5.File(file_name, 'r', swmr=True) as fl:
        t = fl['fields/t'][:]
        nt = len(t)
        # print(nt)

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

    P2_frac_scan[i] = np.mean(P2_frac_t)
    E_frac_scan[i]= np.mean(E_frac_t)
    G_frac_scan[i] = np.mean(G_frac_t)
    Q_scan[i] = np.mean(Q_t)

    P2_frac_scan_err[i] = np.std(P2_frac_t)
    E_frac_scan_err[i] = np.std(E_frac_t)
    G_frac_scan_err[i] = np.std(G_frac_t)
    Q_scan_err[i] = np.std(Q_t)
    # print(f'E_frac_scan_err[{i}] = {E_frac_scan_err[i]}')

#%% Calculate and plot quantities vs time

# Plot P2 fraction vs kapt
plt.figure(figsize=(8,6))
plt.errorbar(kapt_vals, P2_frac_scan, yerr=P2_frac_scan_err, marker='o', linestyle='-', markersize=10, label = '$P_{ZF}^2/P^2$',
             elinewidth=2, capthick=1, capsize=4)
plt.xlabel('$\\kappa_T$')
plt.ylabel('$P_{ZF}^2/P^2$')
plt.title('$P_{ZF}^2/P^2$ vs $\\kappa_T$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(datadir+'zonal_P2_frac_vs_kapt.png',dpi=600)
plt.show()

# Plot total energy fraction vs kapt
plt.figure(figsize=(8,6))
plt.errorbar(kapt_vals, E_frac_scan, yerr=E_frac_scan_err, marker='o', linestyle='-', markersize=10, label = '$\\mathcal{E}_{ZF}/\\mathcal{E}$',
             elinewidth=2, capthick=1, capsize=4)
plt.xlabel('$\\kappa_T$')
plt.ylabel('$\\mathcal{E}_{ZF}/\\mathcal{E}$')
plt.title('$\\mathcal{E}_{ZF}/\\mathcal{E}$ vs $\\kappa_T$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(datadir+'zonal_energy_frac_vs_kapt.png',dpi=600)
plt.show()

# Plot generalized energy fraction vs kapt
plt.figure(figsize=(8,6))
plt.errorbar(kapt_vals, G_frac_scan, yerr=G_frac_scan_err, marker='o', linestyle='-', markersize=10, label = '$\\mathcal{G}_{ZF}/\\mathcal{G}$',
             elinewidth=2, capthick=1, capsize=4)
plt.xlabel('$\\kappa_T$')
plt.ylabel('$\\mathcal{G}_{ZF}/\\mathcal{G}$')
plt.title('$\\mathcal{G}_{ZF}/\\mathcal{G}$ fraction vs $\\kappa_T$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(datadir+'zonal_generalized_energy_frac_vs_kapt.png',dpi=600)
plt.show()

# Plot Q vs kapt
plt.figure(figsize=(8,6))
plt.errorbar(kapt_vals, Q_scan, yerr=Q_scan_err, marker='o', linestyle='-', markersize=10, label = '$\\mathcal{Q}$',
             elinewidth=2, capthick=1, capsize=4)
plt.xlabel('$\\kappa_T$')
plt.ylabel('$\\mathcal{Q}$')
plt.title('$\\mathcal{Q}$ vs $\\kappa_T$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(datadir+'Q_vs_kapt.png',dpi=600)
plt.show()
