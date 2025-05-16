import numpy as np
# from modules.mlsarray import slicelist
import matplotlib.pyplot as plt

def symmetrize_y_axis(axes):
    y_max = np.abs(axes.get_ylim()).max()
    axes.set_ylim(ymin=-y_max, ymax=y_max)

def irft2_g(uk, Nx, Ny):
    Nxh = int(Nx/2)
    u = np.zeros((Npx, int(Npy/2)+1), dtype=complex)
    u[:Nxh,:Nxh] = uk[:Nxh,:Nxh]
    u[-Nxh+1:,:Nxh] = uk[-Nxh+1:,:Nxh]
    return np.fft.irfft2(u, norm='forward')

def rft2_g(u, Nx, Ny):
    Nxh = int(Nx/2)
    uk = np.zeros((Nx, int(Ny/2)+1), dtype=complex)
    yk = np.fft.rfft2(u, norm='forward')
    uk[:Nxh,:-1] = yk[:Nxh,:int(Ny/2)]
    uk[-1:-Nxh:-1,:-1] = yk[-1:-Nxh:-1,:int(Ny/2)]
    uk[0,0] = 0
    return uk

def irft_g(vk, Npx):
    Nxh = int(Npx/3)
    v = np.zeros(int(Npx/2)+1, dtype=complex)
    v[:Nxh] = vk[:Nxh]
    return np.fft.irfft(v, norm='forward')

def ubar(uk, Npx, Npy, Nx, Ny, sl):
    slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
    Nxh = int(Npx/3)
    vk = np.zeros(int(Npx/2)+1, dtype=complex)
    vk[1:Nxh] = uk[slbar]
   
    return np.fft.irfft(vk, norm='forward')
    