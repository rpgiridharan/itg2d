#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np,irftnp,rftnp
from modules.gamma import gam_max   
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 3  
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.width'] = 1.5 
plt.rcParams['ytick.minor.width'] = 1.5 

#%% Load the HDF5 file
datadir = 'data/'
file_name = datadir+'out_kapt_0_36_chi_0_1_H_1_0_em3.h5'
it = -1
# it=100
with h5.File(file_name, 'r', swmr=True) as fl:
    Omk = np.mean(fl['fields/Omk'][-400:],axis=0)
    Pk = np.mean(fl['fields/Pk'][-400:],axis=0)
    Ombar = np.mean(fl['zonal/Ombar'][-400:],axis=0)
    Pbar = np.mean(fl['zonal/Pbar'][-400:],axis=0)
    t = fl['fields/t'][:]
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]
    kapn = fl['params/kapn'][()]
    kapt = fl['params/kapt'][()]
    kapb = fl['params/kapb'][()]
    chi = fl['params/chi'][()]
    a = fl['params/a'][()]
    b = fl['params/b'][()]
    HP = fl['params/HP'][()]
    HPhi = fl['params/HPhi'][()]

Nx,Ny=2*(Npx//3),2*(Npy//3)  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
slky=np.s_[1:int(Ny/2)-1]
gammax=gam_max(kx,ky,kapn,kapt,kapb,chi,a,b,HP,HPhi,slky)
t=t*gammax

print('kx shape', kx.shape, 'ky shape', ky.shape)
nt = len(t)
print("nt: ", nt)

def oneover(arr):
    result = np.zeros_like(arr)
    np.divide(1.0, arr, out=result, where=arr != 0)
    return result

#%% Plots

print(Omk.shape)

kxl = np.r_[0:int(Nx/2), -int(Nx/2):0]
kyl = np.r_[0:int(Ny/2)+1]
dkx = 2*np.pi / Lx
dky = 2*np.pi / Ly
kx_2d, ky_2d = np.meshgrid(kxl * dkx, kyl * dky, indexing='ij')
kp_2d = np.sqrt(kx_2d**2 + ky_2d**2)
sigk_2d = np.sign(np.abs(ky_2d))
fac_2d = sigk_2d + kp_2d**2

Omk_2d = MLSarray(Nx,Ny)
Omk_2d[sl] = cp.array(Omk)
Omk_2d[-1:-int(Nx/2):-1,0] = Omk_2d[1:int(Nx/2),0].conj()
Omk_2d = Omk_2d.get()

print(kx_2d.shape, Omk_2d.shape)

ek = fac_2d * np.abs(Omk_2d)**2 * oneover(kp_2d**4)
slhalf=[np.s_[0:1,1:int(Ny/4)],np.s_[1:int(Nx/4),:int(Ny/4)],np.s_[-int(Nx/2)+1:-int(Nx/4)+1,1:int(Ny/4)]]
# print('ek shape', ek.shape)

#%% Plot      
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = kx_2d
Y = ky_2d
Z = np.abs(Omk_2d)  # or use ek / np.log(ek+eps) if you prefer energy

surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True, rcount=200, ccount=200)
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('$|\\Omega_k|$')
ax.set_title('$\\mathcal{E}_k(k_x, k_y)$; $\\gamma t = %.1f$' % t[it])
ax.view_init(elev=30, azim=-60)
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
plt.tight_layout()
plt.show()

#%% Plot      
fig = plt.figure()
plt.pcolormesh(kx_2d, ky_2d, np.abs(Omk_2d), cmap='viridis')
# plt.pcolormesh(kx_2d, ky_2d, np.log(ek+np.finfo(float).eps), cmap='viridis')
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.title('$\\mathcal{E}_k(k_x, k_y)$; $\\gamma t = %.1f$' % t[it])
plt.tight_layout()
# if file_name.endswith('out.h5'):
#     plt.savefig(datadir+'energy_spectrum.png', dpi=600)
# else:
#     plt.savefig(datadir+"energy_spectrum_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

# %%
