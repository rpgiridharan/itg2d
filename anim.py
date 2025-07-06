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

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2  

#%% Load the HDF5 file

datadir = "data/"
infl = datadir+'out_kapt_1_2_chi_0_1_D_1_0_em3_H_1_0_em3_case6.h5'
outfl = infl.replace('.h5', '.mp4')

with h5.File(infl, "r", libver='latest', swmr=True) as fl:
    t = fl['fields/t'][:]
    Om = fl['fields/Om'][0]
    P = fl['fields/P'][0]
    Om_last = fl['fields/Om'][-1]
    P_last = fl['fields/P'][-1]

Npx, Npy = Om.shape[-2], Om.shape[-1]
Nx,Ny=2*int(np.floor(Npx/3)),2*int(np.floor(Npy/3))

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
        Om = fl['fields/Om'][j]
        P = fl['fields/P'][j]

    qd[0].set_array(Om.T.ravel())
    qd[1].set_array(P.T.ravel())

    tx.set_text('t=' + str(int(t[j]) * 1.0))
    fig.savefig("_tmpimg_folder/tmpout%04i" % (j + nt0) + ".png", dpi=600)
comm.Barrier()

if comm.rank == 0:
    with h5.File(infl, "r", libver='latest', swmr=True) as fl:
        Om = fl['fields/Om'][0]
        P = fl['fields/P'][0]

    qd[0].set_array(Om.T.ravel())
    qd[1].set_array(P.T.ravel())
    tx.set_text('')

    fig.savefig("_tmpimg_folder/tmpout%04i" % (0) + ".png", dpi=200)
    for j in range(1, nt0):
        os.system("cp _tmpimg_folder/tmpout%04i" % (0) + ".png _tmpimg_folder/tmpout%04i" % (j) + ".png")
    
    os.system("ffmpeg -framerate 30 -y -i _tmpimg_folder/tmpout%04d.png -c:v libx264 -pix_fmt yuv420p -vf fps=30 " + outfl)
    shutil.rmtree("_tmpimg_folder")