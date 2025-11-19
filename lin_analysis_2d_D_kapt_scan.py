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
    kapn,kapt,kapb,tau,D,HPhi,HP = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','tau','D','HPhi','HP']
    ]
    kpsq = kx**2 + ky**2
    kpsq = torch.where(kpsq==0, 1e-10, kpsq)
    iv=1
        
    sigk = ky>0
    fac=tau*sigk+kpsq
    lm=torch.zeros(kx.shape+(2,2),dtype=torch.complex64)
    lm[:,:,0,0]=-1j*D*kpsq-1j*sigk*HP/kpsq**3
    lm[:,:,0,1]=(kapn+kapt)*ky
    lm[:,:,1,0]=-kapb*ky/fac
    lm[:,:,1,1]=(kapn*ky-(kapn+kapt)*ky*kpsq-1j*D*kpsq)/fac-1j*sigk*HPhi/kpsq**3

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
kapn=0.2 #rho_i/L_n
kapb=0.02 #2*rho_i/L_B
D=0*0.1
H0=0*1e-5
base_pars={'kapn':kapn,
      'kapb':kapb,
      'tau':1.,#Ti/Te
      'D':D,
      'HPhi':H0,
      'HP':H0}

kapt_vals=np.round(np.arange(0.0,1.8,0.01), 2)
n_kapt=len(kapt_vals)

datadir='data_linear/'
os.makedirs(datadir, exist_ok=True)

#%% Compute

# Create datasets
with h5py.File(datadir + 'gammax_vals_kapt_scan_itg2d_D.h5', 'w') as fl:
    fl.create_dataset('gammax_vals', shape=(n_kapt,), dtype=np.float64)
    fl.create_dataset('kapt_vals', data=kapt_vals, dtype=np.float64)
    fl.create_dataset('kapn', data=kapn, dtype=np.float64)
    fl.create_dataset('kapb', data=kapb, dtype=np.float64)
    
for i in range(len(kapt_vals)):
    base_pars['kapt']=kapt_vals[i] #rho_i/L_T

    print(f'Computing for kapt={kapt_vals[i]}')

    # Compute om
    om=linfreq(base_pars,kx,ky)
    omr=om.real[:,:,0]
    gam=om.imag[:,:,0]

    # Store gammax
    with h5py.File(datadir + 'gammax_vals_kapt_scan_itg2d_D.h5', 'a', libver='latest') as fl:
        # fl.swmr_mode = True
        fl['gammax_vals'][i] = np.max(gam)
        fl.flush()
    gc.collect()