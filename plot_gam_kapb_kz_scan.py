#%% Import modules
import gc
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch 
import h5py

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 3  
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.width'] = 1.5 
plt.rcParams['ytick.minor.width'] = 1.5 

#%% Define Functions

def init_kspace_grid(Nx,Ny,Lx,Ly):
    dkx=2*np.pi/Lx
    dky=2*np.pi/Ly
    kxl=np.r_[np.arange(0,Nx//2),np.arange(-Nx//2,0)]*dkx
    kyl=np.r_[np.arange(0,Ny//2+1)]*dky
    kx,ky=np.meshgrid(kxl,kyl,indexing='ij')
    return kx,ky

def init_linmats(pars,kx,ky):    
    # Initializing the linear matrices
    kapn,kapt,kapb,tau,chi,a,b,s,kz,HPhi,HP,HV,nuPhi,nuP,nuV = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','tau','chi','a','b','s','kz','HPhi','HP','HV','nuPhi','nuP','nuV']
    ]
    kz = torch.ones_like(kx) * kz
    kpsq = kx**2 + ky**2
    kpsq[kpsq==0] = 1e-10
    kzsq = kz**2
    kzsq[kzsq==0] = 1e-10
        
    sigk = ky>0
    fac=tau+kpsq
    lm=torch.zeros(kx.shape+(3,3),dtype=torch.complex64)
    lm[:,:,0,0]=-1j*chi*kpsq-1j*nuP*kzsq**2-1j*sigk*HPhi/kpsq**3
    lm[:,:,0,1]=(5/3)*kz
    lm[:,:,0,2]=(kapn+kapt)*ky
    lm[:,:,1,0]=kz
    lm[:,:,1,1]=-1j*s*chi*kpsq-1j*nuV*kzsq**2-1j*sigk*HV/kpsq**3
    lm[:,:,1,2]=kz
    lm[:,:,2,0]=(-kapb*ky+1j*chi*kpsq**2*b)/fac
    lm[:,:,2,1]=kz/fac
    lm[:,:,2,2]=(kapn*ky-(kapn+kapt)*ky*kpsq-kapb*ky-1j*chi*kpsq**2*a)/fac-1j*nuPhi*kzsq**2-1j*sigk*HPhi/kpsq**3

    return lm

def linfreq(pars, kx, ky):
    lm = init_linmats(pars, torch.from_numpy(kx), torch.from_numpy(ky)).cuda()
    # print(lm.device)
    w = torch.linalg.eigvals(lm)
    iw = torch.argsort(-w.imag, -1)
    lam = torch.gather(w, -1, iw).cpu().numpy()
    # vi = torch.gather(v, -1, iw.unsqueeze(-2).expand_as(v)).cpu().numpy()
    del lm, w, iw
    torch.cuda.empty_cache()
    return lam

#%% Initialize

datadir='data_linear/'
os.makedirs(datadir, exist_ok=True)

# Load datasets
with h5py.File(datadir + 'gammax_vals_kapb_kz_scan_itg2d3c.h5', 'r') as fl:
    gammax_kapb_kz = fl['gammax_vals'][:]
    kapb_vals = fl['kapb_vals'][:]
    kz_vals = fl['kz_vals'][:]
    kapt = fl['kapt'][()]

#%% Colormesh of gam(kapt,kz)

Kapb, Kz = np.meshgrid(kapb_vals, kz_vals)
plt.figure()
plt.pcolormesh(Kapb, Kz, gammax_kapb_kz.T, vmax=1.0, vmin=-1.0, cmap='seismic', rasterized=True, shading='auto')
plt.xlabel('$\\kappa_B$')
plt.ylabel('$k_z$')
plt.title(f"$\\gamma_{{max}}$ for $\\kappa_T$={kapt:.2f}")
plt.colorbar()
plt.savefig(datadir + 'gammax_kapb_kz_itg2d3c.png', dpi=600)
plt.show()
del gammax_kapb_kz