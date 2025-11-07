#%% Import libraries

import numpy as np
import cupy as cp
import h5py as h5
from modules.mlsarray import Slicelist,init_kgrid
from modules.mlsarray import irft2 as original_irft2, rft2 as original_rft2, irft as original_irft, rft as original_rft
from modules.gamma_2d3c import gam_max 
from modules.gensolver import Gensolver,save_data
from modules.basics import round_to_nsig, format_exp
from functools import partial
import os

#%% Parameters

Npx,Npy=512,512
Lx,Ly=32*np.pi,32*np.pi
# kapt_vals=np.arange(0.3,1.6,0.1) 
kapt_vals=np.array([0.3])
kapb=0.05
a=9.0/40.0
b=67.0/160.0
chi=0.1
s=0.9

Nx,Ny=2*(Npx//3),2*(Npy//3)
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
kx,ky=init_kgrid(sl,Lx,Ly)
kpsq=kx**2+ky**2
Nk=kx.size
slky=np.s_[:int(Ny/2)-1]

#%% Functions

irft2 = partial(original_irft2,Npx=Npx,Npy=Npy,Nx=Nx,sl=sl)
rft2 = partial(original_rft2,sl=sl)
irft = partial(original_irft,Npx=Npx,Nx=Nx)
rft = partial(original_rft,Nx=Nx)

def init_fields(kx,ky,w=10.0,A=1e-6):
    Phik=A*cp.exp(-kx**2/2/w**2-ky**2/2/w**2)*cp.array(np.exp(1j*2*np.pi*np.random.rand(kx.size).reshape(kx.shape)))
    Pk=A*cp.exp(-kx**2/2/w**2-ky**2/2/w**2)*cp.array(np.exp(1j*2*np.pi*np.random.rand(kx.size).reshape(kx.shape)))
    Vk=A*cp.exp(-kx**2/2/w**2-ky**2/2/w**2)*cp.array(np.exp(1j*2*np.pi*np.random.rand(kx.size).reshape(kx.shape)))

    Phik[slbar]=0
    Pk[slbar]=0
    Vk[slbar]=0
    zk=np.hstack((Phik,Pk,Vk))
    return zk

def fsavecb(t,y,flag):
    zk=y.view(dtype=complex)
    Phik,Pk,Vk=zk[:Nk],zk[Nk:2*Nk],zk[2*Nk:]
    Omk=-kpsq*Phik
    vy=irft2(1j*kx*Phik) 
    Om=irft2(Omk)
    P=irft2(Pk)
    V=irft2(Vk)
    if flag=='fields':
        save_data(fl,'fields',ext_flag=True,Omk=Omk.get(),Pk=Pk.get(),Vk=Vk.get(),t=t)
    elif flag=='zonal':
        vbar=cp.mean(vy,1)
        Ombar=cp.mean(Om,1)
        Pbar=cp.mean(P,1)
        Vbar=cp.mean(V,1)
        save_data(fl,'zonal',ext_flag=True,vbar=vbar.get(),Ombar=Ombar.get(),Pbar=Pbar.get(),Vbar=Vbar.get(),t=t)
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
    Phik=zk[:Nk]
    Pk=zk[Nk:2*Nk]
    vx=irft2(-1j*ky*Phik)
    P=irft2(Pk)
    Q=np.mean(vx*P)
    Ktot = np.sum(kpsq*np.abs(Phik)**2)
    Kbar = np.sum((kx[slbar]*np.abs(Phik[slbar]))**2)
    print(f'Ktot={Ktot:.3g}, Kbar/Ktot={Kbar/Ktot*100:.3g}%, Q={Q.get():.3g}')

def rhs_itg(t,y):
    zk=y.view(dtype=complex)
    dzkdt=cp.zeros_like(zk)
    Phik,Pk,Vk=zk[:Nk],zk[Nk:2*Nk],zk[2*Nk:]

    dPhikdt,dPkdt,dVkdt=dzkdt[:Nk],dzkdt[Nk:2*Nk],dzkdt[2*Nk:]
    dxphi=irft2(1j*kx*Phik)
    dyphi=irft2(1j*ky*Phik)
    dxP=irft2(1j*kx*Pk)
    dyP=irft2(1j*ky*Pk)
    dxV=irft2(1j*kx*Vk)
    dyV=irft2(1j*ky*Vk)
    sigk=cp.sign(ky)
    fac=sigk+kpsq
    nOmg=irft2(fac*Phik)

    dPhikdt[:]=-1j*kz*sigk*Vk/fac-1j*ky*kapn*Phik/fac+1j*ky*(kapn+kapt)*kpsq*Phik/fac+1j*ky*kapb*Pk/fac-chi*kpsq**2*(a*Phik-b*Pk)/fac
    dPkdt[:]=-(5/3)*1j*kz*sigk*Vk-1j*ky*(kapn+kapt)*Phik-chi*kpsq*Pk
    dVkdt[:]=-1j*kz*sigk*(Pk+Phik)-s*chi*kpsq*Vk

    # dPhikdt[:]+=(1j*kx*rft2(dyphi*nOmg)-1j*ky*rft2(dxphi*nOmg))/fac
    # dPhikdt[:]+= (kx**2*rft2(dxphi*dyP) - ky**2*rft2(dyphi*dxP) + kx*ky*rft2(dyphi*dyP - dxphi*dxP))/fac

    nl_term1_num = 1j*kx*rft2(dyphi*nOmg)-1j*ky*rft2(dxphi*nOmg)
    dPhikdt[:] += nl_term1_num / fac
    nl_term2_num = kx**2*rft2(dxphi*dyP) - ky**2*rft2(dyphi*dxP) + kx*ky*rft2(dyphi*dyP - dxphi*dxP)
    dPhikdt[:] += nl_term2_num / fac

    dPkdt[:]+=rft2(dyphi*dxP-dxphi*dyP)
    dVkdt[:]+=rft2(dyphi*dxV-dxphi*dyV)
    return dzkdt.view(dtype=float)

#%% Run the simulation    

output_dir = "data_2d3c_sweep/"
os.makedirs(output_dir, exist_ok=True)

H0=0*1e-3
HP=H0
HV=H0
HPhi=H0

sim_t0 = 0.0
wecontinue = True
zk = None

for i, kapt_val in enumerate(kapt_vals):
    kapt=round(kapt_val,3)
    kapn=round(kapt/3,3)
    kz=round(0.05*gam_max(kx,ky,kapn,kapt,kapb,chi,a,b,s,0.0,0.0,0.0,0.0,slky)/gam_max(kx,ky,0.4,1.2,kapb,chi,a,b,s,0.0,0.0,0.0,0.0,slky),4)

    filename = output_dir + f'out_2d3c_sweep_kapt_{str(kapt).replace(".","_")}_chi_{str(chi).replace(".","_")}_kz_{str(kz).replace(".","_")}.h5'

    resume_this_step=False
    skip_this_step=False
    t_start=sim_t0
    stored_t1=None

    if wecontinue and os.path.exists(filename):
        with h5.File(filename,'r') as existing:
            has_last=('last' in existing and 'zk' in existing['last'] and 't' in existing['last'])
            if 'data' in existing and 't1' in existing['data']:
                stored_t1=float(existing['data/t1'][()])
            if has_last:
                last_t=float(existing['last/t'][()])
                if (stored_t1 is not None) and (np.isclose(last_t,stored_t1,rtol=1e-6,atol=1e-8) or last_t>stored_t1):
                    skip_this_step=True
                    zk=cp.array(existing['last/zk'][()])
                else:
                    resume_this_step=True
                    t_start=last_t
                    zk=cp.array(existing['last/zk'][()])

    if skip_this_step:
        print(f'  Skipping sweep step {i+1}/{len(kapt_vals)}; existing run complete for kapt={kapt}')
        continue

    if zk is None:
        zk=init_fields(kx,ky)
        print(f'  Initialized fields for first kapt value: {kapt}')
    elif resume_this_step:
        print(f'  Resuming run for kapt={kapt} from t={t_start:.3g}')
    else:
        print(f'  Using final state from previous run as initial condition for kapt: {kapt}')

    dtshow=0.1
    gammax=gam_max(kx,ky,kapn,kapt,kapb,chi,a,b,s,kz,HPhi,HP,HV,slky)
    dtstep,dtsavecb=round_to_nsig(0.00275/gammax,1),round_to_nsig(0.0275/gammax,1)
    t1=round(300/gammax,0) #100/gammax #1200/gammax
    rtol,atol=1e-8,1e-10

    if resume_this_step and (np.isclose(t_start,t1,rtol=1e-6,atol=1e-8) or t_start>t1):
        print(f'  Skipping sweep step {i+1}/{len(kapt_vals)}; checkpoint already at t={t1} for kapt={kapt}')
        continue

    file_mode='r+' if resume_this_step and os.path.exists(filename) else 'w'
    fl=h5.File(filename,file_mode,libver='latest')
    fl.swmr_mode = True

    save_data(fl,'data',ext_flag=False,kx=kx.get(),ky=ky.get(),t0=sim_t0,t1=t1)
    save_data(fl,'params',ext_flag=False,Npx=Npx,Npy=Npy,Lx=Lx,Ly=Ly,kapn=kapn,kapt=kapt,kapb=kapb,chi=chi,a=a,b=b,HP=HP,HPhi=HPhi)

    fsave = [partial(fsavecb,flag='fields'), partial(fsavecb,flag='zonal'), partial(fsavecb,flag='fluxes')]
    dtsave=[10*dtsavecb,dtsavecb,dtsavecb]
    r=Gensolver('cupy_ivp.DOP853',rhs_itg,t_start,zk.view(dtype=float),t1,fsave=fsave,fshow=fshowcb,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,dense=False,rtol=rtol,atol=atol)
    r.run()
    fl.flush()
    zk = cp.array(fl['last/zk'][()])

    fl.close()
    print(f'  Completed sweep step {i+1}/{len(kapt_vals)} for kapt={kapt}')
