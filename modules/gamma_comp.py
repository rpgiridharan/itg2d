import numpy as np
import cupy as cp
import torch

def init_linmats(kx,ky,pars):    
    # Initializing the linear matrices
    kapn,kapt,kapb,tau,chi,a,b,HPhi,HP = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','tau','chi','a','b','HPhi','HP']
    ]
    kpsq = kx**2 + ky**2
    # kpsq = torch.where(kpsq==0, 1e-10, kpsq)
        
    sigk = ky>0
    fac=sigk+kpsq
    lm=torch.zeros(kx.shape+(2,2),dtype=torch.complex64)
    lm[:,0,0]=-2*(5/3)*kapb*ky-1j*chi*kpsq-1j*sigk*HP/kpsq**3
    lm[:,0,1]=(kapn+kapt+(tau-1)*kapb)*ky
    lm[:,1,0]=(-kapb*ky+1j*chi*kpsq**2*b)/fac
    lm[:,1,1]=(-(kapb-kapn)*ky-(kapn+kapt)*ky*kpsq-1j*chi*kpsq**2*a)/fac-1j*sigk*HPhi/kpsq**3

    return lm

def linfreq(kx, ky, pars):
    lm = init_linmats(torch.from_numpy(kx), torch.from_numpy(ky), pars).cuda()
    # print(lm.device)
    w = torch.linalg.eigvals(lm)
    iw = torch.argsort(-w.imag, -1)
    lam = torch.gather(w, -1, iw).cpu().numpy()
    del lm, w, iw
    torch.cuda.empty_cache()
    return lam

def gam_max(kx, ky, kapn, kapt, kapb, chi, a, b, HPhi, HP, slky):
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
        'HPhi':HPhi,
        'HP':HP}

    om=linfreq(kx,ky,base_pars)
    gamky=om.imag[slky,0]
    return np.max(gamky)

def ky_max(kx, ky, kapn, kapt, kapb, chi, a, b, HPhi, HP, slky):
    if isinstance(ky, cp.ndarray):
        kx = kx.get()
        ky = ky.get()

    base_pars={'kapn':kapn,
        'kapt':kapt,
        'kapb':kapb,
        'chi':chi,
        'a':a,
        'b':b,
        'Hphi':HPhi,
        'Ht':HP}

    om=linfreq(base_pars,kx,ky)
    gamky=om.imag[slky,0]
    return ky[slky,np.argmax(gamky)]
