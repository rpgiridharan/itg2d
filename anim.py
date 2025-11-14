#%% Import Librarires
import sys
import os
import shutil
import numpy as np
import h5py as h5
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
from mpl_toolkits.axes_grid1 import make_axes_locatable
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np,irftnp,rftnp

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

datadir = "data/"
infl = datadir+'out_kapt_1_2_D_0_1_H_1_0_em3.h5'
# datadir = "data_2d3c/"
# infl = datadir+'out_2d3c_kapt_1_2_chi_0_1_kz_0_01.h5'

outfl = infl.replace('.h5', '.mp4')

with h5.File(infl, "r", libver='latest', swmr=True) as fl:
    t = fl['fields/t'][:]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]

with h5.File(infl, "r", libver='latest', swmr=True) as fl:
    Om = irft2np(fl['fields/Omk'][0],Npx,Npy,Nx,sl)
    P = irft2np(fl['fields/Pk'][0],Npx,Npy,Nx,sl)
    Om_last = irft2np(fl['fields/Omk'][-1],Npx,Npy,Nx,sl)
    P_last = irft2np(fl['fields/Pk'][-1],Npx,Npy,Nx,sl)

Om_max_last = np.max(np.abs(Om_last)) 
P_max_last = np.max(np.abs(P_last))

#%% Plot the data

w, h = 9.6, 5.4
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(w, h))
qd = []
qd.append(ax[0].pcolormesh(Om.T, cmap='seismic', rasterized=True, vmin=-Om_max_last, vmax=Om_max_last, shading='auto'))
qd.append(ax[1].pcolormesh(P.T, cmap='seismic', rasterized=True, vmin=-P_max_last, vmax=P_max_last, shading='auto'))
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')

divider0 = make_axes_locatable(ax[0])
divider1 = make_axes_locatable(ax[1])
cax0 = divider0.append_axes("right", size="5%", pad=0.05)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(qd[0], cax=cax0)
cbar2 = fig.colorbar(qd[1], cax=cax1)

ax[0].set_title('$\\Omega$', pad=-1)
ax[1].set_title('$P$', pad=-1)
ax[0].tick_params('y', labelleft=False)
ax[0].tick_params('x', labelbottom=False)
ax[1].tick_params('y', labelleft=False)
ax[1].tick_params('x', labelbottom=False)

plt.subplots_adjust(wspace=0.2, hspace=0.2)

nt0 = 10
nt = t.shape[0]

ax[0].axis('off')
ax[1].axis('off')

tx = fig.text(0.515, 0.925, "t=0", ha='center')
if (comm.rank == 0):
    lt = np.arange(0,nt,1)
    lt_loc = np.array_split(lt, comm.size)
    if not os.path.exists('_tmpimg_folder'):
        os.makedirs('_tmpimg_folder')
else:
    lt_loc = None
lt_loc = comm.scatter(lt_loc, root=0)

for j in lt_loc:
    print(j)
    with h5.File(infl, "r", libver='latest', swmr=True) as fl:
        Om = irft2np(fl['fields/Omk'][j],Npx,Npy,Nx,sl)
        P = irft2np(fl['fields/Pk'][j],Npx,Npy,Nx,sl)

    qd[0].set_array(Om.T.ravel())
    qd[1].set_array(P.T.ravel())

    tx.set_text('t=' + str(int(t[j]) * 1.0))
    fig.savefig("_tmpimg_folder/tmpout%04i" % (j + nt0) + ".png", dpi=600)
comm.Barrier()

if comm.rank == 0:
    with h5.File(infl, "r", libver='latest', swmr=True) as fl:
        Om = irft2np(fl['fields/Omk'][0],Npx,Npy,Nx,sl)
        P = irft2np(fl['fields/Pk'][0],Npx,Npy,Nx,sl)

    qd[0].set_array(Om.T.ravel())
    qd[1].set_array(P.T.ravel())
    tx.set_text('')

    fig.savefig("_tmpimg_folder/tmpout%04i" % (0) + ".png", dpi=200)
    for j in range(1, nt0):
        os.system("cp _tmpimg_folder/tmpout%04i" % (0) + ".png _tmpimg_folder/tmpout%04i" % (j) + ".png")
    
    os.system("ffmpeg -framerate 30 -y -i _tmpimg_folder/tmpout%04d.png -c:v libx264 -pix_fmt yuv420p -vf fps=30 " + outfl)
    shutil.rmtree("_tmpimg_folder")