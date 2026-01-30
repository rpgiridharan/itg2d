#%% Import modules
import gc
import os
import numpy as np
import matplotlib.pyplot as plt
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

def compute_discriminant(kpsq, kapn, kapt, kapb):
    A = kapn + kapt
    return (kapn - A * kpsq)**2 - 4.0 * kapb * (1.0 + kpsq) * A

# This script intentionally does not compute gamma; only discriminant

#%% Initialize

Npx,Npy=4096,4096
Nx,Ny=2*int(Npx/3),2*int(Npy/3)
# Lx,Ly=32*np.pi,32*np.pi #sim for 512x512
Lx,Ly=256*np.pi,256*np.pi 
kx,ky=init_kspace_grid(Nx,Ny,Lx,Ly)
kapt=0.4 #rho_i/L_T >0.2
kapn=0.2 #rho_i/L_n
kapb=0.04 #2*rho_i/L_B
D=0*0.1 #0.1
H0= 0 # 
base_pars={'kapn':kapn,
      'kapt':kapt,
      'kapb':kapb,
      'D':D,
      'HPhi':H0,
      'HP':H0}

kapn_vals=np.round(np.arange(-1.0,1.0,0.05),2)
kapt_vals=np.round(np.arange(-1.0,1.0,0.05),2)

n_kapn=len(kapn_vals)
n_kapt=len(kapt_vals)

datadir='data_linear/'
os.makedirs(datadir, exist_ok=True)
file_name = datadir + f'discmax_vals_itg2d_kapn_kapt_scan_kapb_{str(kapb).replace(".", "_")}.h5'

#%% Compute

# Create datasets

with h5py.File(file_name, 'w') as fl:
    shape = (n_kapn, n_kapt)
    fl.create_dataset('kapt_vals', data=kapt_vals, dtype=np.float64)
    fl.create_dataset('kapn_vals', data=kapn_vals, dtype=np.float64)
    fl.create_dataset('kapb', data=kapb, dtype=np.float64)
    # Only save max(-Δ) which is zero when Δ≥0 everywhere
    fl.create_dataset('negdiscmax_vals', shape, dtype=np.float64)
    
for i in range(len(kapn_vals)):
    base_pars['kapn']=kapn_vals[i] #rho_i/L_n
    for j in range(len(kapt_vals)):
        base_pars['kapt']=kapt_vals[j] #rho_i/L_T
        print(f'Computing for kapn={kapn_vals[i]}, kapt={kapt_vals[j]}, kapb={kapb}')

        # Compute and store max(-Δ) over k-grid (zero if no negative Δ)
        with h5py.File(file_name, 'a', libver='latest') as fl:
            kpsq = kx**2 + ky**2
            disc = compute_discriminant(kpsq, base_pars['kapn'], base_pars['kapt'], base_pars['kapb'])
            negdiscmax = float(np.max(-disc))
            fl['negdiscmax_vals'][i, j] = negdiscmax
            fl.flush()
        gc.collect()