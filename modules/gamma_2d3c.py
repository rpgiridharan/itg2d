import numpy as np
import cupy as cp
import torch

def init_linmats(kx,ky,pars):    
    # Initializing the linear matrices
    kapn,kapt,kapb,tau,chi,a,b,s,kz,HP,HV,HPhi = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','tau','chi','a','b','s','kz','HP','HV','HPhi']
    ]
    kpsq = kx**2 + ky**2
    # kpsq = torch.where(kpsq==0, 1e-10, kpsq)
        
    sigk = ky>0
    fac=sigk+kpsq
    lm=torch.zeros(kx.shape+(3,3),dtype=torch.complex64)
    lm[:,0,0]=-1j*chi*kpsq-1j*sigk*HP/kpsq**3
    lm[:,0,1]=(5/3)*kz
    lm[:,0,2]=(kapn+kapt)*ky
    lm[:,1,0]=kz
    lm[:,1,1]=-1j*s*chi*kpsq-1j*sigk*HV/kpsq**3
    lm[:,1,2]=kz
    lm[:,2,0]=(-kapb*ky+1j*chi*kpsq**2*b)/fac
    lm[:,2,1]=kz/fac
    lm[:,2,2]=(kapn*ky-(kapn+kapt)*ky*kpsq-kapb*ky-1j*chi*kpsq**2*a)/fac-1j*sigk*HPhi/kpsq**3

    return lm

def linfreq(kx, ky, pars):
    lm = init_linmats(torch.from_numpy(kx), torch.from_numpy(ky), pars).cuda()
    w = torch.linalg.eigvals(lm)
    iw = torch.argsort(-w.imag, -1)
    lam = torch.gather(w, -1, iw).cpu().numpy()
    del lm, w, iw
    torch.cuda.empty_cache()
    return lam

def gam_max(kx, ky, kapn, kapt, kapb, chi, a, b, s, kz, HP, HV, HPhi, slky):
    if isinstance(ky, cp.ndarray):
        kx = kx.get()
        ky = ky.get()
        
    base_pars={'kapn':kapn,
        'kapt':kapt,
        'kapb':kapb,
        'tau':1.,#Ti/Te
        'chi':chi,
        'a':a,
        'b':b,
        's':s,
        'kz':kz,
        'HP':HP,
        'HV':HV,
        'HPhi':HPhi}

    om=linfreq(kx,ky,base_pars)
    gamky=om.imag[slky,0]
    return np.max(gamky)

def ky_max(kx, ky, kapn, kapt, kapb, chi, a, b, s, kz, HP, HV, HPhi, slky):
    if isinstance(ky, cp.ndarray):
        kx = kx.get()
        ky = ky.get()

    base_pars={'kapn':kapn,
        'kapt':kapt,
        'kapb':kapb,
        'chi':chi,
        'a':a,
        'b':b,
        's':s,
        'kz':kz,
        'HP':HP,
        'HV':HV,
        'HPhi':HPhi}

    om=linfreq(base_pars,kx,ky)
    gamky=om.imag[slky,0]
    return ky[slky,np.argmax(gamky)]
