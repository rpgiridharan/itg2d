#%% Import modules
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch 

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 3  
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.width'] = 1.5 
plt.rcParams['ytick.minor.width'] = 1.5 


#%% Define Functions

def init_kspace_grid(Nx,Ny,Lx,Ly):
    dkx=2*np.pi/Lx
    dky=2*np.pi/Ly
    kxl=np.r_[np.arange(0,Nx//2),np.arange(-Nx//2,0)]*dkx
    kyl=np.r_[np.arange(0,Ny//2+1)]*dky
    kx,ky=np.meshgrid(kxl,kyl,indexing='ij')
    return kx,ky

def init_linmats(pars,kx,ky):    
    # Initializing the linear matrices
    kapn,kapt,kapb,eps,tau,chi,a,b,Hphi,Ht = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','eps','tau','chi','a','b','Hphi','Ht']
    ]
    sqre = torch.sqrt(eps)
    kpsq = kx**2 + ky**2
    kpsq = torch.where(kpsq==0, 1e-10, kpsq)
        
    sigk = ky>0
    fac=tau+kpsq
    lm=torch.zeros(kx.shape+(2,2),dtype=torch.complex64)
    lm[:,:,0,0]=-1j*chi*kpsq-1j*sigk*Ht/kpsq**3
    lm[:,:,0,1]=(kapn+kapt)*ky
    lm[:,:,1,0]=(-kapb*ky+1j*chi*kpsq**2*b)/fac
    lm[:,:,1,1]=(kapn*ky-(kapn+kapt)*ky*kpsq-kapb*ky-1j*chi*kpsq**2*a)/fac-1j*sigk*Hphi/kpsq**3

    return lm

def linfreq(pars, kx, ky):
    lm = init_linmats(pars, torch.from_numpy(kx), torch.from_numpy(ky)).cuda()
    # print(lm.device)
    w = torch.linalg.eigvals(lm)
    iw = torch.argsort(-w.imag, -1)
    lam = torch.gather(w, -1, iw).cpu().numpy()
    # vi = torch.gather(v, -1, iw.unsqueeze(-2).expand_as(v)).cpu().numpy()
    del lm, w, iw
    torch.cuda.empty_cache()
    return lam

#%% Initialize

Npx,Npy=512,512
Nx,Ny=2*int(Npx/3),2*int(Npy/3)
Lx,Ly=32*np.pi,32*np.pi
kx,ky=init_kspace_grid(Nx,Ny,Lx,Ly)
kapn=0*0.5 #rho_i/L_n
kapt=2.0 #rho_i/L_T
kapb=1.0 #2*rho_i/L_B
chi=0.1
a=9.0/40.0
b=67.0/160.0
H0=5e-12
base_pars={'kapn':kapn,
      'kapt':kapt,
      'kapb':kapb,
      'eps':0.1,#trapped e fraction
      'tau':1.,#Ti/Te
      'chi':chi,
      'a':a,
      'b':b,
      'Hphi':H0,
      'Ht':H0}

#%% Compute om

om=linfreq(base_pars,kx,ky)
omr=om.real[:,:,0]
gam=om.imag[:,:,0]

#%% Compute max quantities

print('gammax:',np.max(gam),'1/gammax:',1/np.max(gam))
print('max index:', np.unravel_index(np.argmax(gam[:,:]), gam.shape))
print('max kx:', kx[np.unravel_index(np.argmax(gam[:,:]), gam.shape)])
print('max ky:', ky[np.unravel_index(np.argmax(gam[:,:]), gam.shape)])

ind_kxmax = np.argmax(gam, axis=0,keepdims=True) 
gam_kxmax = np.take_along_axis(gam, ind_kxmax, axis=0).squeeze(axis=0)  #select gam at the max kx index
omr_kxmax = np.take_along_axis(omr, ind_kxmax, axis=0).squeeze(axis=0) 
gam_kx0 = gam[0,:]
omr_kx0 = omr[0,:]

#%% Plots
   
plt.figure()
# slx=slice(None,int(Nx/32),1) 
slx=slice(None,int(Nx/8),int((Nx/8)/8)) #9 kx points
plt.plot(ky[slx,:int(Ny/8)].T,gam[slx,:int(Ny/8)].T,'.-')
plt.plot(ky[0,:int(Ny/8)],0*ky[0,:int(Ny/8)]**2,'k--')
# plt.plot(ky[0,:int(Ny/8)],-a*chi*ky[0,:int(Ny/8)]**2-D0*ky[0,:int(Ny/8)]**4,'k--')
plt.legend(['$k_x='+str(l)+'$' for l in kx[slx,0]]+['$-a\\chi k_y^2$'])
plt.xlabel('$k_y$')
plt.ylabel('$\\gamma(k_y)$')
plt.title('$\\gamma(k_{xi},k_y)$ vs $k_x$ for diff $k_x$')
plt.tight_layout()
plt.savefig('data/gam_vs_ky_kxvals_itg2d.png',dpi=600)
plt.show()

kymax_kx= np.take_along_axis(ky[:int(Nx/4),:],np.argmax(gam[:int(Nx/4),:],axis=1,keepdims=True),axis=1).squeeze(axis=1)
plt.figure()
plt.plot(kx[:int(Nx/4),0],kymax_kx,'.-')
plt.xlabel('$k_x$')
plt.ylabel('$k_{y,max}$')
plt.title('$k_{y,max}$ vs $k_x$')
plt.tight_layout()
plt.savefig('data/ky_vs_kx_itg2d.png',dpi=600)
# plt.show()

#%% colormesh of gam and omr

kx_shifted = np.fft.fftshift(kx[:,:],axes=0)
ky_shifted = np.fft.fftshift(ky[:,:],axes=0)

plt.figure()
plt.pcolormesh(kx_shifted, ky_shifted, np.fft.fftshift(gam, axes=0),vmax=0.3,vmin=-0.3,cmap='seismic', rasterized=True)
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.title('$\\gamma(k_x,k_y)$')
plt.colorbar()
plt.tight_layout()
plt.savefig('data/gamkxky_itg2d.png',dpi=600)
# plt.show()

# plt.figure()
# plt.pcolormesh(kx_shifted, ky_shifted, np.fft.fftshift(omr, axes=0),vmax=0.2,vmin=-0.2,cmap='seismic', rasterized=True)
# plt.xlabel('$k_x$')
# plt.ylabel('$k_y$')
# plt.title('$\\omega_r(k_x,k_y) = \\omega_r(k_x,k_y)$')
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('data/omrkxky_itg2d.png',dpi=600)
# plt.show()
