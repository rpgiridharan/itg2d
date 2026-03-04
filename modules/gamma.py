import numpy as np
import cupy as cp
import torch

def init_linmats(kx,ky,pars):    
    # Initializing the linear matrices
    kapn,kapt,kapb,D,H = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','D','H']
    ]
    kpsq = kx**2 + ky**2
        
    sigk = ky>0
    fac=sigk+kpsq
    lm=torch.zeros(kx.shape+(2,2),dtype=torch.complex64)
    lm[:,0,0]=-1j*D*kpsq-1j*sigk*H/kpsq**2
    lm[:,0,1]=(kapn+kapt)*ky
    lm[:,1,0]=-kapb*ky/fac
    lm[:,1,1]=(-(kapb-kapn)*ky-(kapn+kapt)*ky*kpsq)/fac-1j*D*kpsq-1j*sigk*H/kpsq**2

    return lm

def linfreq(kx, ky, pars):
    lm = init_linmats(torch.from_numpy(kx), torch.from_numpy(ky), pars).cuda()
    w = torch.linalg.eigvals(lm)
    iw = torch.argsort(-w.imag, -1)
    lam = torch.gather(w, -1, iw).cpu().numpy()
    del lm, w, iw
    torch.cuda.empty_cache()
    return lam

def gam_max(kx, ky, kapn, kapt, kapb, D, H):
    if isinstance(ky, cp.ndarray):
        kx = kx.get()
        ky = ky.get()
        
    base_pars={'kapn':kapn,
        'kapt':kapt,
        'kapb':kapb,
        'D':D,
        'H':H}

    om=linfreq(kx,ky,base_pars)
    gam=om.imag[:,0]
    return np.max(gam)

def ky_max(kx, ky, kapn, kapt, kapb, D, H):
    if isinstance(ky, cp.ndarray):
        kx = kx.get()
        ky = ky.get()

    base_pars={'kapn':kapn,
        'kapt':kapt,
        'kapb':kapb,
        'D':D,
        'H':H}

    om=linfreq(kx, ky, base_pars)
    gam=om.imag[:,0]
    return ky[np.argmax(gam)]

def Dturb(kx, ky, kapn, kapt, kapb, D, H):
    if isinstance(ky, cp.ndarray):
        kx = kx.get()
        ky = ky.get()

    base_pars={'kapn':kapn,
        'kapt':kapt,
        'kapb':kapb,
        'D':D,
        'H':H}

    om=linfreq(kx, ky, base_pars)
    gam=om.imag[:,0]
    return gam/(kx**2+ky**2)
