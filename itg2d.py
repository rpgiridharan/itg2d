#%% Import libraries

import numpy as np
import cupy as cp
import h5py as h5
from modules.mlsarray import Slicelist,init_kgrid
from modules.mlsarray import irft2 as original_irft2, rft2 as original_rft2, irft as original_irft, rft as original_rft
from modules.gamma import gam_max   
from modules.gensolver import Gensolver,save_data
from functools import partial
import os

#%% Parameters

Npx,Npy=512,512
Lx,Ly=32*np.pi,32*np.pi
kapn=0.0
kapt=1.2#0.36 
kapb=1.0
a=9.0/40.0
b=67.0/160.0
chi=0.1
DPhi=1e-3
DP=1e-3
HPhi=1e-3
HP=1e-3

Nx,Ny=2*(Npx//3),2*(Npy//3)
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
kx,ky=init_kgrid(sl,Lx,Ly)
kpsq=kx**2+ky**2
Nk=kx.size
ky0=ky[:Ny/2-1]

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
output_dir = "data/"
os.makedirs(output_dir, exist_ok=True)
filename = output_dir + f'out_kapt_{str(kapt).replace(".", "_")}_chi_{str(chi).replace(".", "_")}_D_{format_exp(DPhi)}_H_{format_exp(HPhi)}_debug_l.h5'

dtshow=0.1
gammax=round(gam_max(ky0,kapt),6)
dtstep,dtsavecb=round(0.00275/gammax,3),round(0.0275/gammax,3)
t0,t1=0.0,int(round(100/gammax)/dtstep)*dtstep #3000/gammax
rtol,atol=1e-9,1e-11
wecontinue=False

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
    vy=irft2(1j*kx*Phik) 
    Om=irft2(-kpsq*Phik)
    P=irft2(Pk)
    if flag=='fields':
        save_data(fl,'fields',ext_flag=True,Om=Om.get(),P=P.get(),t=t)
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

    print(f"Time: {t}")
    print(f"kpsq: type={type(kpsq)}, shape={kpsq.shape}, dtype={kpsq.dtype}, is_contiguous={kpsq.flags.c_contiguous}, has_nan={cp.any(cp.isnan(kpsq))}, has_inf={cp.any(cp.isinf(kpsq))}")
    print(f"Phik: type={type(Phik)}, shape={Phik.shape}, dtype={Phik.dtype}, is_contiguous={Phik.flags.c_contiguous}, has_nan={cp.any(cp.isnan(Phik))}, has_inf={cp.any(cp.isinf(Phik))}")
    print(f"ky: type={type(ky)}, shape={ky.shape}, dtype={ky.dtype}, is_contiguous={ky.flags.c_contiguous}, has_nan={cp.any(cp.isnan(ky))}, has_inf={cp.any(cp.isinf(ky))}")
    print(f"fac: type={type(fac)}, shape={fac.shape}, dtype={fac.dtype}, is_contiguous={fac.flags.c_contiguous}, has_nan={cp.any(cp.isnan(fac))}, has_inf={cp.any(cp.isinf(fac))}, has_zero={cp.any(fac == 0)}")

    dPhikdt[:]=1j*ky*(kapb-kapn)*Phik/fac+1j*ky*(kapn+kapt)*kpsq*Phik/fac+1j*ky*kapb*Pk/fac-chi*kpsq**2*(a*Phik-b*Pk)/fac-sigk*(DPhi*kpsq**2*Phik+HPhi/(kpsq**3)*Phik) 
    dPkdt[:]=-1j*ky*(kapn+kapt)*Phik-chi*kpsq*Pk-sigk*(DP*kpsq**2*Pk+HP/(kpsq**3)*Pk) 

    dPhikdt[:]+=(1j*kx*rft2(dyphi*nOmg)-1j*ky*rft2(dxphi*nOmg))/fac
    dPhikdt[:]+= (kx**2*rft2(dxphi*dyP) - ky**2*rft2(dyphi*dxP) + kx*ky*rft2(dyphi*dyP - dxphi*dxP))/fac
    dPkdt[:]+=rft2(dyphi*dxP-dxphi*dyP)
    return dzkdt.view(dtype=float)

def rhs_itg_debug(t,y):
    zk=y.view(dtype=complex)
    dzkdt=cp.zeros_like(zk)
    # It's often safer to work with copies if zk might be modified elsewhere,
    # or if the solver relies on y not changing during rhs_itg.
    # However, for performance, views are common. Let's assume current usage is intended.
    Phik,Pk=zk[:Nk],zk[Nk:]

    dPhikdt_view,dPkdt_view=dzkdt[:Nk],dzkdt[Nk:] # Views for assignment

    dxphi=irft2(1j*kx*Phik)
    dyphi=irft2(1j*ky*Phik)
    dxP=irft2(1j*kx*Pk)
    dyP=irft2(1j*ky*Pk)
    sigk=cp.sign(ky)
    fac=sigk+kpsq # kpsq is global
    nOmg=irft2(fac*Phik)

    # Your diagnostic prints here (they are good)
    print(f"Time: {t}")
    print(f"kpsq: type={type(kpsq)}, shape={kpsq.shape}, dtype={kpsq.dtype}, is_contiguous={kpsq.flags.c_contiguous}, has_nan={cp.any(cp.isnan(kpsq))}, has_inf={cp.any(cp.isinf(kpsq))}")
    print(f"Phik: type={type(Phik)}, shape={Phik.shape}, dtype={Phik.dtype}, is_contiguous={Phik.flags.c_contiguous}, has_nan={cp.any(cp.isnan(Phik))}, has_inf={cp.any(cp.isinf(Phik))}")
    print(f"ky: type={type(ky)}, shape={ky.shape}, dtype={ky.dtype}, is_contiguous={ky.flags.c_contiguous}, has_nan={cp.any(cp.isnan(ky))}, has_inf={cp.any(cp.isinf(ky))}")
    print(f"fac: type={type(fac)}, shape={fac.shape}, dtype={fac.dtype}, is_contiguous={fac.flags.c_contiguous}, has_nan={cp.any(cp.isnan(fac))}, has_inf={cp.any(cp.isinf(fac))}, has_zero={cp.any(fac == 0)}")

    # --- Break down dPhikdt calculation ---
    # Term 1
    term_A1 = (kapb - kapn) * Phik
    term_A = 1j * ky * term_A1 / fac
    print(f"DEBUG: term_A computed. Has NaN: {cp.any(cp.isnan(term_A))}, Has Inf: {cp.any(cp.isinf(term_A))}")

    # Term 2 (original suspect for SystemError)
    term_B1 = (kapn + kapt) * kpsq
    term_B2 = term_B1 * Phik
    term_B = 1j * ky * term_B2 / fac
    print(f"DEBUG: term_B computed. Has NaN: {cp.any(cp.isnan(term_B))}, Has Inf: {cp.any(cp.isinf(term_B))}")

    # Term 3
    term_C1 = kapb * Pk
    term_C = 1j * ky * term_C1 / fac
    print(f"DEBUG: term_C computed. Has NaN: {cp.any(cp.isnan(term_C))}, Has Inf: {cp.any(cp.isinf(term_C))}")

    # Term 4
    term_D1_chi_part = a * Phik - b * Pk
    term_D2_chi_part = kpsq**2 * term_D1_chi_part
    term_D = -chi * term_D2_chi_part / fac
    print(f"DEBUG: term_D computed. Has NaN: {cp.any(cp.isnan(term_D))}, Has Inf: {cp.any(cp.isinf(term_D))}")

    # Term 5 (Hyperviscosity)
    HPhi_contribution = HPhi * Phik / (kpsq**3)
    term_E1_hypervisc_part = DPhi * kpsq**2 * Phik
    term_E2_hypervisc_part = term_E1_hypervisc_part + HPhi_contribution
    term_E = -sigk * term_E2_hypervisc_part
    print(f"DEBUG: term_E computed. Has NaN: {cp.any(cp.isnan(term_E))}, Has Inf: {cp.any(cp.isinf(term_E))}")

    dPhikdt_view[:] = term_A + term_B + term_C + term_D + term_E
    print(f"DEBUG: dPhikdt initial assignment done. Has NaN: {cp.any(cp.isnan(dPhikdt_view))}, Has Inf: {cp.any(cp.isinf(dPhikdt_view))}")


    # --- Break down dPkdt calculation ---
    pk_term_A = -1j * ky * (kapn + kapt) * Phik

    pk_term_B = -chi * kpsq * Pk

    HP_contribution = HP * Pk / (kpsq**3)
    pk_term_C1_hypervisc_part = DP * kpsq**2 * Pk
    pk_term_C2_hypervisc_part = pk_term_C1_hypervisc_part + HP_contribution
    pk_term_C = -sigk * pk_term_C2_hypervisc_part
    
    dPkdt_view[:] = pk_term_A + pk_term_B + pk_term_C
    print(f"DEBUG: dPkdt initial assignment done. Has NaN: {cp.any(cp.isnan(dPkdt_view))}, Has Inf: {cp.any(cp.isinf(dPkdt_view))}")


    # --- Nonlinear terms (additive) ---
    nl_term1_num = 1j*kx*rft2(dyphi*nOmg)-1j*ky*rft2(dxphi*nOmg)
    dPhikdt_view[:] += nl_term1_num / fac
    print(f"DEBUG: dPhikdt after NL1. Has NaN: {cp.any(cp.isnan(dPhikdt_view))}, Has Inf: {cp.any(cp.isinf(dPhikdt_view))}")

    nl_term2_num = kx**2*rft2(dxphi*dyP) - ky**2*rft2(dyphi*dxP) + kx*ky*rft2(dyphi*dyP - dxphi*dxP)
    dPhikdt_view[:] += nl_term2_num / fac
    print(f"DEBUG: dPhikdt after NL2. Has NaN: {cp.any(cp.isnan(dPhikdt_view))}, Has Inf: {cp.any(cp.isinf(dPhikdt_view))}")
    
    dPkdt_view[:] += rft2(dyphi*dxP-dxphi*dyP)
    print(f"DEBUG: dPkdt after NL. Has NaN: {cp.any(cp.isnan(dPkdt_view))}, Has Inf: {cp.any(cp.isinf(dPkdt_view))}")
    
    return dzkdt.view(dtype=float)

def rhs_itg_debug_l(t,y):
    zk=y.view(dtype=complex)
    dzkdt=cp.zeros_like(zk)
    # It's often safer to work with copies if zk might be modified elsewhere,
    # or if the solver relies on y not changing during rhs_itg.
    # However, for performance, views are common. Let's assume current usage is intended.
    Phik,Pk=zk[:Nk],zk[Nk:]

    dPhikdt_view,dPkdt_view=dzkdt[:Nk],dzkdt[Nk:] # Views for assignment

    dxphi=irft2(1j*kx*Phik)
    dyphi=irft2(1j*ky*Phik)
    dxP=irft2(1j*kx*Pk)
    dyP=irft2(1j*ky*Pk)
    sigk=cp.sign(ky)
    fac=sigk+kpsq # kpsq is global
    nOmg=irft2(fac*Phik)

    # Your diagnostic prints here (they are good)
    print(f"Time: {t}")
    print(f"kpsq: type={type(kpsq)}, shape={kpsq.shape}, dtype={kpsq.dtype}, is_contiguous={kpsq.flags.c_contiguous}, has_nan={cp.any(cp.isnan(kpsq))}, has_inf={cp.any(cp.isinf(kpsq))}")
    print(f"Phik: type={type(Phik)}, shape={Phik.shape}, dtype={Phik.dtype}, is_contiguous={Phik.flags.c_contiguous}, has_nan={cp.any(cp.isnan(Phik))}, has_inf={cp.any(cp.isinf(Phik))}")
    print(f"ky: type={type(ky)}, shape={ky.shape}, dtype={ky.dtype}, is_contiguous={ky.flags.c_contiguous}, has_nan={cp.any(cp.isnan(ky))}, has_inf={cp.any(cp.isinf(ky))}")
    print(f"fac: type={type(fac)}, shape={fac.shape}, dtype={fac.dtype}, is_contiguous={fac.flags.c_contiguous}, has_nan={cp.any(cp.isnan(fac))}, has_inf={cp.any(cp.isinf(fac))}, has_zero={cp.any(fac == 0)}")

    # --- Break down dPhikdt calculation ---
    # Term 1
    term_A1 = (kapb - kapn) * Phik
    term_A = 1j * ky * term_A1 / fac
    print(f"DEBUG: term_A computed. Has NaN: {cp.any(cp.isnan(term_A))}, Has Inf: {cp.any(cp.isinf(term_A))}")

    # Term 2 (original suspect for SystemError)
    term_B1 = (kapn + kapt) * kpsq
    term_B2 = term_B1 * Phik
    term_B = 1j * ky * term_B2 / fac
    print(f"DEBUG: term_B computed. Has NaN: {cp.any(cp.isnan(term_B))}, Has Inf: {cp.any(cp.isinf(term_B))}")

    # Term 3
    term_C1 = kapb * Pk
    term_C = 1j * ky * term_C1 / fac
    print(f"DEBUG: term_C computed. Has NaN: {cp.any(cp.isnan(term_C))}, Has Inf: {cp.any(cp.isinf(term_C))}")

    # Term 4
    term_D1_chi_part = a * Phik - b * Pk
    term_D2_chi_part = kpsq**2 * term_D1_chi_part
    term_D = -chi * term_D2_chi_part / fac
    print(f"DEBUG: term_D computed. Has NaN: {cp.any(cp.isnan(term_D))}, Has Inf: {cp.any(cp.isinf(term_D))}")

    # Term 5 (Hyperviscosity)
    HPhi_contribution = HPhi * Phik / (kpsq**3)
    term_E1_hypervisc_part = DPhi * kpsq**2 * Phik
    term_E2_hypervisc_part = term_E1_hypervisc_part + HPhi_contribution
    term_E = -sigk * term_E2_hypervisc_part
    print(f"DEBUG: term_E computed. Has NaN: {cp.any(cp.isnan(term_E))}, Has Inf: {cp.any(cp.isinf(term_E))}")

    dPhikdt_view[:] = term_A + term_B + term_C + term_D + term_E
    print(f"DEBUG: dPhikdt initial assignment done. Has NaN: {cp.any(cp.isnan(dPhikdt_view))}, Has Inf: {cp.any(cp.isinf(dPhikdt_view))}")


    # --- Break down dPkdt calculation ---
    pk_term_A = -1j * ky * (kapn + kapt) * Phik

    pk_term_B = -chi * kpsq * Pk

    HP_contribution = HP * Pk / (kpsq**3)
    pk_term_C1_hypervisc_part = DP * kpsq**2 * Pk
    pk_term_C2_hypervisc_part = pk_term_C1_hypervisc_part + HP_contribution
    pk_term_C = -sigk * pk_term_C2_hypervisc_part
    
    dPkdt_view[:] = pk_term_A + pk_term_B + pk_term_C
    print(f"DEBUG: dPkdt initial assignment done. Has NaN: {cp.any(cp.isnan(dPkdt_view))}, Has Inf: {cp.any(cp.isinf(dPkdt_view))}")


    # --- Nonlinear terms (additive) --- difference being /fac is in the same line
    dPhikdt_view[:]+=(1j*kx*rft2(dyphi*nOmg)-1j*ky*rft2(dxphi*nOmg))/fac
    dPhikdt_view[:]+= (kx**2*rft2(dxphi*dyP) - ky**2*rft2(dyphi*dxP) + kx*ky*rft2(dyphi*dyP - dxphi*dxP))/fac
    dPkdt_view[:]+=rft2(dyphi*dxP-dxphi*dyP)
    
    return dzkdt.view(dtype=float)


#%% Run the simulation    

print(f'chi={chi}, kapn={kapn}, kapt={kapt}, kapb={kapb}')

if(wecontinue):
    fl=h5.File(filename,'r+',libver='latest')
    fl.swmr_mode = True
    zk=fl['last/zk'][()]
    t0=fl['last/t'][()]
else:
    fl=h5.File(filename,'w',libver='latest')
    fl.swmr_mode = True
    zk=init_fields(kx,ky)
    save_data(fl,'data',ext_flag=False,kx=kx.get(),ky=ky.get(),t0=t0,t1=t1)
    save_data(fl,'params',ext_flag=False,Npx=Npx,Npy=Npy,Lx=Lx,Ly=Ly,kapn=kapn,kapt=kapt,kapb=kapb,chi=chi,a=a,b=b,HP=HP,HPhi=HPhi)

fsave = [partial(fsavecb,flag='fields'), partial(fsavecb,flag='zonal'), partial(fsavecb,flag='fluxes')]
dtsave=[10*dtsavecb,dtsavecb,dtsavecb]
r=Gensolver('cupy_ivp.DOP853',rhs_itg_debug_l,t0,zk.view(dtype=float),t1,fsave=fsave,fshow=fshowcb,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,dense=False,rtol=rtol,atol=atol)
r.run()
fl.close()
