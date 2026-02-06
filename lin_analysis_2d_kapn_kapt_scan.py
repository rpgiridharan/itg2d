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
    kapn,kapt,kapb,D,H = [
        torch.tensor(pars[l], device=kx.device, dtype=kx.dtype) for l in ['kapn','kapt','kapb','D','H']
    ]
    kpsq = kx**2 + ky**2
    kpsq = torch.where(kpsq==0, 1e-10, kpsq)
        
    sigk = ky>0
    fac=sigk+kpsq
    lm=torch.zeros(kx.shape+(2,2),dtype=torch.complex64)
    lm[:,:,0,0]=-1j*sigk*D*kpsq-1j*sigk*H/kpsq**2
    lm[:,:,0,1]=(kapn+kapt)*ky
    lm[:,:,1,0]=-kapb*ky/fac
    lm[:,:,1,1]=(kapn*ky-(kapn+kapt)*ky*kpsq)/fac-1j*sigk*D*kpsq-1j*sigk*H/kpsq**2

    return lm

def eigvals_2x2(lm):
    a = lm[..., 0, 0]
    b = lm[..., 0, 1]
    c = lm[..., 1, 0]
    d = lm[..., 1, 1]
    tr = a + d
    disc = (a - d) ** 2 / 4 + b * c
    root = torch.sqrt(disc)
    lam1 = tr / 2 + root
    lam2 = tr / 2 - root
    return torch.stack((lam1, lam2), dim=-1)

def linfreq(pars, kx, ky):    
    lm = init_linmats(pars, torch.from_numpy(kx), torch.from_numpy(ky)).cuda()
    # w = torch.linalg.eigvals(lm)
    w = eigvals_2x2(lm)
    iw = torch.argsort(-w.imag, -1)
    lam = torch.gather(w, -1, iw).cpu().numpy()
    del lm, w, iw
    torch.cuda.empty_cache()
    return lam

#%% Initialize

Npx,Npy=4096,4096
Nx,Ny=2*int(Npx/3),2*int(Npy/3)
# Lx,Ly=32*np.pi,32*np.pi #sim for 512x512
Lx,Ly=256*np.pi,256*np.pi 
kx,ky=init_kspace_grid(Nx,Ny,Lx,Ly)
kapt=0.4 #rho_i/L_T >0.2
kapn=0.2 #rho_i/L_n
kapb=0.02 #2*rho_i/L_B
D=0.1 #0.1
H=1e-6 # 
base_pars={'kapn':kapn,
      'kapt':kapt,
      'kapb':kapb,
      'D':D,
      'H':H}

kapn_vals=np.round(np.arange(-0.5,1.0,0.02),2)
kapt_vals=np.round(np.arange(-0.5,1.0,0.02),2)

n_kapn=len(kapn_vals)
n_kapt=len(kapt_vals)

datadir='data_linear/'
os.makedirs(datadir, exist_ok=True)
file_name = datadir + f'lin_kapn_kapt_scan_kapb_{str(kapb).replace(".", "_")}_itg2d.h5'
# file_name = datadir + f'lin_kapn_kapt_scan_kapb_{str(kapb).replace(".", "_")}_itg2d_wo_FLR.h5'

def one_over(x):
    out = np.zeros_like(x)
    return np.divide(1.0, x, out=out, where=x != 0)

#%% Compute

# Create datasets

with h5py.File(file_name, 'w') as fl:
    shape = (n_kapn, n_kapt)
    fl.create_dataset('gammax_vals', shape, dtype=np.float64)
    fl.create_dataset('Dturbmax_vals', shape, dtype=np.float64)
    fl.create_dataset('kapt_vals', data=kapt_vals, dtype=np.float64)
    fl.create_dataset('kapn_vals', data=kapn_vals, dtype=np.float64)
    fl.create_dataset('kapb', data=kapb, dtype=np.float64)
    
for i in range(len(kapn_vals)):
    base_pars['kapn']=kapn_vals[i] #rho_i/L_n
    for j in range(len(kapt_vals)):
        base_pars['kapt']=kapt_vals[j] #rho_i/L_T
        print(f'Computing for kapn={kapn_vals[i]}, kapt={kapt_vals[j]}, kapb={kapb}')

        # Compute om
        om=linfreq(base_pars,kx,ky)
        omr=om.real[:,:,0]
        gam=om.imag[:,:,0]
        Dturb=gam*one_over(kx**2+ky**2)

        # Store gammax
        with h5py.File(file_name, 'a', libver='latest') as fl:
            # fl.swmr_mode = True
            fl['gammax_vals'][i, j] = np.max(gam)
            fl['Dturbmax_vals'][i, j] = np.max(Dturb)
            fl.flush()
        gc.collect()