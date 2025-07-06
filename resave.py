import os
import h5py as h5
import numpy as np
import cupy as cp
from modules.mlsarray import Slicelist, rft2

# Directories
src_dir = 'data/'
dst_dir = 'data_resave/'
os.makedirs(dst_dir, exist_ok=True)

# List all .h5 files in source directory
# files = [f for f in os.listdir(src_dir) if f.endswith('.h5')]
files = ['out_kapt_1_2_chi_0_1_D_0_0_e0_H_1_0_em3.h5']  
for fname in files:
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)
    print(f'Processing {fname}...')

    with h5.File(src_path, 'r') as fsrc, h5.File(dst_path, 'w') as fdst:
        # Copy all groups except 'fields'
        for key in fsrc.keys():
            if key != 'fields':
                fsrc.copy(key, fdst)

        # Prepare for fields group
        grp_src = fsrc['fields']
        t = grp_src['t'][:]
        Om = grp_src['Om'][:]
        P = grp_src['P'][:]

        # Get params for Slicelist
        Npx = fsrc['params/Npx'][()]
        Npy = fsrc['params/Npy'][()]
        Nx, Ny = 2*Npx//3, 2*Npy//3
        sl = Slicelist(Nx, Ny)

        Nk = cp.hstack(rft2(cp.asarray(Om[0]), sl)).size

        # Compute Omk and Pk
        Omk = np.empty((Om.shape[0], Nk), dtype=np.complex128)
        Pk = np.empty((P.shape[0], Nk), dtype=np.complex128)
        for i in range(Om.shape[0]):
            Omk[i] = cp.asnumpy(rft2(cp.asarray(Om[i]), sl))
            Pk[i] = cp.asnumpy(rft2(cp.asarray(P[i]), sl))

        # Save new fields group
        grp_dst = fdst.create_group('fields')
        grp_dst.create_dataset('Omk', data=Omk)
        grp_dst.create_dataset('Pk', data=Pk)
        grp_dst.create_dataset('t', data=t)