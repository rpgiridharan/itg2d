#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np
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

def K(Omk, kpsq):
    ''' Returns the total kinetic energy of the system'''
    E = np.sum(np.abs(-Omk)**2/kpsq).item()
    return np.real(E)

def K_ZF(Omk, kpsq, slbar):
    ''' Returns the zonal kinetic energy of the system'''
    E_ZF = np.sum(np.abs(-Omk[slbar])**2/kpsq[slbar]).item()
    return np.real(E_ZF)

def W(Omk):
    ''' Returns the total enstrophy of the system'''
    Wtemp = np.sum(np.abs(Omk)**2).item()
    return Wtemp
    
def W_ZF(Omk, slbar):
    ''' Returns the zonal enstrophy of the system'''
    W_ZFtemp = np.sum(np.abs(Omk[slbar])**2).item()
    return W_ZFtemp

def Kfrac(Omk, kpsq, slbar):
    ''' Returns the total kinetic energy of the system'''
    E = np.sum(np.abs(-Omk)**2/kpsq[None,:], axis=-1)
    E_ZF = np.sum(-np.abs(Omk[slbar])**2/kpsq[None,slbar],axis=-1)
    return E_ZF/E

def Wfrac(Omk, slbar):
    ''' Returns the total enstrophy of the system'''
    W = np.sum(np.abs(-Omk)**2, axis=-1)
    W_ZF = np.sum(np.abs(Omk[slbar])**2, axis=-1)
    return W_ZF/W

def Gam(Omk, kpsq, slbar):
    ''' Returns the particle flux of the system'''
    Phik = -Omk/kpsq
    Gam = np.sum(Omk[slbar] * np.conj(Omk[slbar])).item()
    return Gam

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

def format_exp(d):
    dstr = f"{d:.1e}"
    base, exp = dstr.split("e")
    base = base.replace(".", "_")
    if "-" in exp:
        exp = exp.replace("-", "")
        prefix = "em"
    else:
        prefix = "e"
    exp = str(int(exp))
    return f"{base}_{prefix}{exp}"

#%% Define the quantities to be plotted

datadir = 'data_sweep/'
kapt_vals = np.arange(0.3,1.5,0.1)

E_frac_scan = np.zeros(kapt_vals.shape)
E_frac_scan_err = np.zeros(kapt_vals.shape)
W_frac_scan = np.zeros(kapt_vals.shape)
W_frac_scan_err = np.zeros(kapt_vals.shape)
Q_scan = np.zeros(kapt_vals.shape)
Q_scan_err = np.zeros(kapt_vals.shape)

for i,kapt in enumerate(kapt_vals):
    kapt = round(kapt, 3)
    print(f'Processing kappa_T = {kapt}')

    pattern = datadir + f'out_sweep_kapt_{str(kapt).replace(".", "_")}_*.h5'
    files = glob.glob(pattern)
    if not files:
        print(f"No file found for kappa_T = {kapt}")
        continue
    file_name = files[0] 
    print(file_name)

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

        E_frac_t = np.zeros(int(nt/2))
        W_frac_t = np.zeros(int(nt/2))
        for it in range(int(nt/2)):
            Omk = fl['fields/Omk'][it+nt//2]
            Pk = fl['fields/Pk'][it+nt//2]
            E_frac_t[it] = K_ZF(Omk, kpsq, slbar) / K(Omk, kpsq)
            W_frac_t[it] = W_ZF(Omk, slbar) / W(Omk)
        Q_t = np.zeros(int(ntf/2))
        for it in range(int(ntf/2)):
            Q = fl['fluxes/Q'][it+ntf//2]
            Q_t[it] = np.mean(Q)

    E_frac_scan[i]= np.mean(E_frac_t)
    W_frac_scan[i] = np.mean(W_frac_t)
    Q_scan[i] = np.mean(Q_t)

    E_frac_scan_err[i] = np.std(E_frac_t)
    W_frac_scan_err[i] = np.std(W_frac_t)
    Q_scan_err[i] = np.std(Q_t)
    # print(f'E_frac_scan_err[{i}] = {E_frac_scan_err[i]}')

#%% Calculate and plot quantities vs time

# Plot kinetic energy vs kapt
plt.figure(figsize=(8,6))
plt.errorbar(kapt_vals, E_frac_scan, yerr=E_frac_scan_err, marker='o', linestyle='-', markersize=10, label = '$\\mathcal{E}_{ZF}/\\mathcal{E}$',
             elinewidth=2, capthick=1, capsize=4)
plt.xlabel('$\\kappa_T$')
plt.ylabel('$\\mathcal{E}_{ZF}/\\mathcal{E}$')
plt.title('$\\mathcal{E}_{ZF}/\\mathcal{E}$ vs $\\kappa_T$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(datadir+'zonal_energy_frac_vs_kapt_sweep.pdf',dpi=100)
plt.show()

# # Plot enstrophy vs kapt
# plt.figure(figsize=(8,6))
# plt.errorbar(kapt_vals, W_frac_scan, yerr=W_frac_scan_err, marker='o', linestyle='-', markersize=10, label = '$\\mathcal{W}_{ZF}/\\mathcal{W}$',
#              elinewidth=2, capthick=1, capsize=4)
# plt.xlabel('$\\kappa_T$')
# plt.ylabel('$\\mathcal{W}_{ZF}/\\mathcal{W}$')
# plt.title('$\\mathcal{W}_{ZF}/\\mathcal{W}$ fraction vs $\\kappa_T$')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(datadir+'zonal_enstrophy_frac_vs_kapt_sweep.pdf',dpi=100)
# plt.show()

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
plt.savefig(datadir+'Q_vs_kapt_sweep.pdf',dpi=100)
plt.show()
