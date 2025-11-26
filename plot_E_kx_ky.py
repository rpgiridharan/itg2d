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

#%% Load the HDF5 file
datadir = 'data/'
kapt=1.0
D=0.1
pattern = datadir + f'out_kapt_{str(kapt).replace(".", "_")}_D_{str(D).replace(".", "_")}*.h5'
files = glob.glob(pattern)
if not files:
    print(f"No file found for kappa_T = {kapt}")
else:
    file_name = files[0]

with h5.File(file_name, 'r', swmr=True) as fl:
    t = fl['fields/t'][:]
    nt = len(t)
    Omk = np.mean(fl['fields/Omk'][-int(nt/2):],axis=0)
    Pk = np.mean(fl['fields/Pk'][-int(nt/2):],axis=0)
    Ombar = np.mean(fl['zonal/Ombar'][-int(nt/2):],axis=0)
    Pbar = np.mean(fl['zonal/Pbar'][-int(nt/2):],axis=0)
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]
    kapn = fl['params/kapn'][()]
    kapt = fl['params/kapt'][()]
    kapb = fl['params/kapb'][()]
    D = fl['params/D'][()]
    HP = fl['params/HP'][()]
    HPhi = fl['params/HPhi'][()]

Nx,Ny=2*(Npx//3),2*(Npy//3)  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
slky=np.s_[1:int(Ny/2)-1]
gammax=gam_max(kx,ky,kapn,kapt,kapb,D,HP,HPhi,slky)
t=t*gammax

print('kx shape', kx.shape, 'ky shape', ky.shape)
print("nt: ", nt)

#%% Functions

def oneover(arr):
    result = np.zeros_like(arr)
    np.divide(1.0, arr, out=result, where=arr != 0)
    return result

def init_kspace_grid(Nx,Ny,Lx,Ly):
    dkx=2*np.pi/Lx
    dky=2*np.pi/Ly
    kxl=np.r_[np.arange(0,Nx//2),np.arange(-Nx//2,0)]*dkx
    kyl=np.r_[np.arange(0,Ny//2+1)]*dky
    kx,ky=np.meshgrid(kxl,kyl,indexing='ij')
    return kx,ky

#%% Plots

print(Omk.shape)

kx,ky=init_kspace_grid(Nx,Ny,Lx,Ly)
kx = np.fft.fftshift(kx,axes=0)
ky = np.fft.fftshift(ky,axes=0)
kp = np.sqrt(kx**2 + ky**2)
sigk_2d = np.sign(np.abs(ky))
fac_2d = sigk_2d + kp**2

Omk_2d = MLSarray(Nx,Ny)
Omk_2d[sl] = cp.array(Omk)
Omk_2d[-1:-int(Nx/2):-1,0] = Omk_2d[1:int(Nx/2),0].conj()
Omk_2d = Omk_2d.get()
Omk_2d = np.fft.fftshift(Omk_2d,axes=0)

ek = fac_2d * np.abs(Omk_2d)**2 * oneover(kp**4)
roi = (slice(int(Nx/4), int(3*Nx/4)), slice(0, int(Ny/4)))

#%% Surface plot of E_kx_ky      

X = kx
Y = ky
Z = np.log(ek+np.finfo(float).eps)  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(kx[roi], ky[roi], Z[roi], cmap='viridis', linewidth=0, antialiased=True, rcount=200, ccount=200)
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('$log(\\mathcal{E}_k)$')
ax.set_title('$log(\\mathcal{E}_k)(k_x, k_y)$')
ax.view_init(elev=30, azim=-60)
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.15)
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'E_kx_ky_surf.png', dpi=600)
else:
    plt.savefig(datadir+"E_kx_ky_surf_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

#%% Colormesh of log(E_kx_ky)      

fig = plt.figure()
plt.pcolormesh(kx[roi], ky[roi], np.log(ek[roi]+np.finfo(float).eps), cmap='viridis', shading='auto')
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.title('$log(\\mathcal{E}_k)(k_x, k_y)$')
plt.colorbar()
plt.tight_layout()
if file_name.endswith('out.h5'):
    plt.savefig(datadir+'E_kx_ky.png', dpi=600)
else:
    plt.savefig(datadir+"E_kx_ky_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()

#%% Colormesh of E_kx_ky 

roi2 = (slice(int(3*Nx/8), int(5*Nx/8)), slice(0, int(Ny/8)))
fig = plt.figure()
plt.pcolormesh(kx[roi2], ky[roi2], ek[roi2], cmap='Blues')
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.title('$\\mathcal{E}_k(k_x, k_y)$')
plt.colorbar()
plt.tight_layout()
# if file_name.endswith('out.h5'):
#     plt.savefig(datadir+'E_kx_ky.png', dpi=600)
# else:
#     plt.savefig(datadir+"E_kx_ky_" + file_name.split('/')[-1].split('out_')[-1].replace('.h5', '.png'), dpi=600)
plt.show()