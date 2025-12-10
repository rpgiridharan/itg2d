import numpy as np
import cupy as cp
import torch

def init_linmats(kx,ky,pars):    
    # Initializing the linear matrices
    kapn,kapt,kapb,D,kz = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','D','kz']
    ]
    kpsq = kx**2 + ky**2
    # kpsq = torch.where(kpsq==0, 1e-10, kpsq)
        
    sigk = ky>0
    fac=sigk+kpsq
    lm=torch.zeros(kx.shape+(3,3),dtype=torch.complex64)
    lm[:,0,0]=-1j*sigk*D*kpsq
    lm[:,0,1]=(5/3)*kz
    lm[:,0,2]=(kapn+kapt)*ky
    lm[:,1,0]=kz
    lm[:,1,1]=-1j*sigk*D*kpsq
    lm[:,1,2]=kz
    lm[:,2,0]=-kapb*ky/fac
    lm[:,2,1]=kz/fac
    lm[:,2,2]=(kapn*ky-(kapn+kapt)*ky*kpsq-kapb*ky)/fac-1j*sigk*D*kpsq

    return lm

def linfreq(kx, ky, pars):
    lm = init_linmats(torch.from_numpy(kx), torch.from_numpy(ky), pars).cuda()
    w = torch.linalg.eigvals(lm)
    iw = torch.argsort(-w.imag, -1)
    lam = torch.gather(w, -1, iw).cpu().numpy()
    del lm, w, iw
    torch.cuda.empty_cache()
    return lam

def gam_max(kx, ky, kapn, kapt, kapb, D, kz):
    if isinstance(ky, cp.ndarray):
        kx = kx.get()
        ky = ky.get()
        
    base_pars={'kapn':kapn,
        'kapt':kapt,
        'kapb':kapb,
        'D':D,
        'kz':kz}

    om=linfreq(kx,ky,base_pars)
    gam=om.imag[:,0]
    return np.max(gam)

def ky_max(kx, ky, kapn, kapt, kapb, D, kz):
    if isinstance(ky, cp.ndarray):
        kx = kx.get()
        ky = ky.get()

    base_pars={'kapn':kapn,
        'kapt':kapt,
        'kapb':kapb,
        'D':D,
        'kz':kz}

    om=linfreq(kx,ky,base_pars)
    gam=om.imag[:,0]
    return ky[np.argmax(gam)]
