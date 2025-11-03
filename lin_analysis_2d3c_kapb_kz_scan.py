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

Npx,Npy=512,512
Nx,Ny=2*int(Npx/3),2*int(Npy/3)
Lx,Ly=32*np.pi,32*np.pi
kx,ky=init_kspace_grid(Nx,Ny,Lx,Ly)
kapt=1.2 #rho_i/L_T
kapn=kapt/3 #rho_i/L_n
chi=0.1
a=9.0/40.0
b=67.0/160.0
s=0.9
H0=0*1e-5
nu0=0*1e-6
base_pars={'kapt':kapt,
    'kapn':kapn,
    'tau':1.,#Ti/Te
    'chi':chi,
    'a':a,
    'b':b,
    's':s,
    'HPhi':H0,
    'HP':H0,
    'HV':H0,
    'nuPhi':nu0,
    'nuP':nu0,
    'nuV':nu0}

kapb_vals=np.round(np.arange(0.005,0.1,0.005), 3)
kz_vals=np.round(np.arange(0.,5.,0.25), 2)

n_kapb=len(kapb_vals)
n_kz=len(kz_vals)

datadir='data_linear/'
os.makedirs(datadir, exist_ok=True)

#%% Compute

# Create datasets
with h5py.File(datadir + 'gammax_vals_kapb_kz_scan_itg2d3c.h5', 'w') as fl:
    shape = (n_kapb, n_kz)
    fl.create_dataset('gammax_vals', shape, dtype=np.float64)
    fl.create_dataset('kapb_vals', data=kapb_vals, dtype=np.float64)
    fl.create_dataset('kz_vals', data=kz_vals, dtype=np.float64)
    fl.create_dataset('kapt', data=kapt, dtype=np.float64)

for i in range(len(kapb_vals)):
    base_pars['kapb']=kapb_vals[i] #rho_i/L_B
    for j in range(len(kz_vals)):
        base_pars['kz']=kz_vals[j]
        print(f'Computing for kapb={kapb_vals[i]}, kz={kz_vals[j]}')

        # Compute om
        om=linfreq(base_pars,kx,ky)
        omr=om.real[:,:,0]
        gam=om.imag[:,:,0]

        # Store gammax
        with h5py.File(datadir + 'gammax_vals_kapb_kz_scan_itg2d3c.h5', 'a', libver='latest') as fl:
            # fl.swmr_mode = True
            fl['gammax_vals'][i, j] = np.max(gam)
            fl.flush()
        gc.collect()