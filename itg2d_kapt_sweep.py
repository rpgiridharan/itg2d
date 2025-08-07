#%% Import libraries

import numpy as np
import cupy as cp
import h5py as h5
from modules.mlsarray import Slicelist,init_kgrid
from modules.mlsarray import irft2 as original_irft2, rft2 as original_rft2, irft as original_irft, rft as original_rft
from modules.gamma import gam_max, gam_kmin   
from modules.gensolver import Gensolver,save_data
from functools import partial
import os

#%% Parameters

Npx,Npy=512,512
Lx,Ly=32*np.pi,32*np.pi
kapn=0.0
kapt_vals=np.arange(0.3,1.6,0.1)  # Scan over kapt values
kapb=1.0
a=9.0/40.0
b=67.0/160.0
chi=0.1

Nx,Ny=2*(Npx//3),2*(Npy//3)
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
kx,ky=init_kgrid(sl,Lx,Ly)
kpsq=kx**2+ky**2
Nk=kx.size
ky0=ky[:Ny/2-1]

#%% Functions

irft2 = partial(original_irft2,Npx=Npx,Npy=Npy,Nx=Nx,sl=sl)
rft2 = partial(original_rft2,sl=sl)
irft = partial(original_irft,Npx=Npx,Nx=Nx)
rft = partial(original_rft,Nx=Nx)

def init_fields(kx,ky,w=10.0,A=1e-6):
    Phik=A*cp.exp(-kx**2/2/w**2-ky**2/2/w**2)*cp.exp(1j*2*np.pi*cp.random.rand(kx.size).reshape(kx.shape))
    Pk=A*cp.exp(-kx**2/2/w**2-ky**2/2/w**2)*cp.exp(1j*2*np.pi*cp.random.rand(kx.size).reshape(kx.shape))
    Phik[slbar]=0
    Pk[slbar]=0
    zk=np.hstack((Phik,Pk))
    return zk

def fsavecb(t,y,flag):
    zk=y.view(dtype=complex)
    Phik,Pk=zk[:Nk],zk[Nk:]
    Omk=-kpsq*Phik
    vy=irft2(1j*kx*Phik) 
    Om=irft2(Omk)
    P=irft2(Pk)
    if flag=='fields':
        save_data(fl,'fields',ext_flag=True,Omk=Omk.get(),Pk=Pk.get(),t=t)
    elif flag=='zonal':
        vbar=cp.mean(vy,1)
        Ombar=cp.mean(Om,1)
        Pbar=cp.mean(P,1)
        save_data(fl,'zonal',ext_flag=True,vbar=vbar.get(),Ombar=Ombar.get(),Pbar=Pbar.get(),t=t)
    elif flag=='fluxes':
        vx=irft2(-1j*ky*Phik) #ExB flow: x comp
        wx=irft2(-1j*ky*Pk) #diamagnetic flow: x comp
        Q=cp.mean(P*vx,1)
        R=cp.mean(vy*vx,1)
        PiP=cp.mean(vy*wx,1)
        save_data(fl,'fluxes',ext_flag=True,Q=Q.get(),R=R.get(),PiP=PiP.get(),t=t)
    save_data(fl,'last',ext_flag=False,zk=zk.get(),t=t)

def fshowcb(t,y):
    zk=y.view(dtype=complex)
    Phik,Pk=zk[:Nk],zk[Nk:]
    vx=irft2(-1j*ky*Phik)
    P=irft2(Pk)
    Q=np.mean(vx*P)
    Ktot = np.sum(kpsq*np.abs(Phik)**2)
    Kbar = np.sum((kx[slbar]*np.abs(Phik[slbar]))**2)
    print(f'Ktot={Ktot:.3g}, Kbar/Ktot={Kbar/Ktot*100:.3g}%, Q={Q.get():.3g}')

def rhs_itg(t,y):
    zk=y.view(dtype=complex)
    dzkdt=cp.zeros_like(zk)
    Phik,Pk=zk[:Nk],zk[Nk:]

    dPhikdt,dPkdt=dzkdt[:Nk],dzkdt[Nk:]
    dxphi=irft2(1j*kx*Phik)
    dyphi=irft2(1j*ky*Phik)
    dxP=irft2(1j*kx*Pk)
    dyP=irft2(1j*ky*Pk)
    sigk=cp.sign(ky)
    fac=sigk+kpsq
    nOmg=irft2(fac*Phik)

    dPhikdt[:]=1j*ky*(kapb-kapn)*Phik/fac+1j*ky*(kapn+kapt)*kpsq*Phik/fac+1j*ky*kapb*Pk/fac-chi*kpsq**2*(a*Phik-b*Pk)/fac-sigk*HPhi/(kpsq**3)*Phik
    dPkdt[:]=-1j*ky*(kapn+kapt)*Phik-chi*kpsq*Pk-sigk*HP/(kpsq**3)*Pk

    dPhikdt[:]+=(1j*kx*rft2(dyphi*nOmg)-1j*ky*rft2(dxphi*nOmg))/fac
    dPhikdt[:]+= (kx**2*rft2(dxphi*dyP) - ky**2*rft2(dyphi*dxP) + kx*ky*rft2(dyphi*dyP - dxphi*dxP))/fac

    nl_term1_num = 1j*kx*rft2(dyphi*nOmg)-1j*ky*rft2(dxphi*nOmg)
    dPhikdt[:] += nl_term1_num / fac
    nl_term2_num = kx**2*rft2(dxphi*dyP) - ky**2*rft2(dyphi*dxP) + kx*ky*rft2(dyphi*dyP - dxphi*dxP)
    dPhikdt[:] += nl_term2_num / fac

    dPkdt[:]+=rft2(dyphi*dxP-dxphi*dyP)
    return dzkdt.view(dtype=float)

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

def round_to_nsig(number, n):
    """Rounds a number to n significant figures."""
    if not np.isfinite(number): # Catches NaN, Inf, -Inf
        return number 
    if number == 0:
        return 0.0
    if n <= 0:
        raise ValueError("Number of significant figures (n) must be positive.")
    
    order_of_magnitude = np.floor(np.log10(np.abs(number)))
    decimals_to_round = int(n - 1 - order_of_magnitude)
    
    return np.round(number, decimals=decimals_to_round)

#%% Run the simulation    

# Initialize zk only once for the first run
zk = None

for i, kapt in enumerate(kapt_vals):
    kapt=round(kapt,3)
    H0 = 1e-3*gam_kmin(ky0,kapt)/gam_kmin(ky0,1.2)
    HPhi=H0
    HP=H0

    output_dir = "data_sweep/"
    os.makedirs(output_dir, exist_ok=True)
    filename = output_dir + f'out_sweep_kapt_{str(kapt).replace(".", "_")}_chi_{str(chi).replace(".", "_")}_H_{format_exp(HPhi)}.h5'

    dtshow=0.1
    gammax=round(gam_max(ky0,kapt),3)
    # dtstep,dtsavecb=round(0.00275/gammax,3),round(0.0275/gammax,3)
    dtstep,dtsavecb=round_to_nsig(0.00275/gammax,1),round_to_nsig(0.0275/gammax,1)
    t0,t1=0.0,round(100/gammax,0) #3000/gammax
    rtol,atol=1e-8,1e-10

    # Create new file for each kapt value
    fl=h5.File(filename,'w',libver='latest')
    fl.swmr_mode = True
    
    # Initialize fields only for the first run, use previous final state for subsequent runs
    if i == 0:
        zk=init_fields(kx,ky)
        print(f'  Initialized fields for first kapt value: {kapt}')
    else:
        print(f'  Using final state from previous run as initial condition for kapt: {kapt}')
    
    save_data(fl,'data',ext_flag=False,kx=kx.get(),ky=ky.get(),t0=t0,t1=t1)
    save_data(fl,'params',ext_flag=False,Npx=Npx,Npy=Npy,Lx=Lx,Ly=Ly,kapn=kapn,kapt=kapt,kapb=kapb,chi=chi,a=a,b=b,HP=HP,HPhi=HPhi)

    fsave = [partial(fsavecb,flag='fields'), partial(fsavecb,flag='zonal'), partial(fsavecb,flag='fluxes')]
    dtsave=[10*dtsavecb,dtsavecb,dtsavecb]
    r=Gensolver('cupy_ivp.DOP853',rhs_itg,t0,zk.view(dtype=float),t1,fsave=fsave,fshow=fshowcb,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,dense=False,rtol=rtol,atol=atol)
    r.run()
    
    # Get the final state for use as initial condition in next run
    zk = cp.array(r.y[:,-1].view(dtype=complex))
    
    fl.close()
    print(f'  Completed sweep step {i+1}/{len(kapt_vals)} for kapt={kapt}')
