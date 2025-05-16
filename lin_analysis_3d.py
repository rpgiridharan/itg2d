#%% Import modules
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch 

#%% Define Functions

def init_kspace_grid(Nx,Ny,Nz,Lx,Ly,Lz):
    dkx=2*np.pi/Lx
    dky=2*np.pi/Ly
    dkz=2*np.pi/Lz
    kxl=np.r_[np.arange(0,Nx//2),np.arange(-Nx//2,0)]*dkx
    kyl=np.r_[np.arange(0,Ny//2),np.arange(-Ny//2,0)]*dky
    kzl=np.r_[np.arange(0,Nz//2+1)]*dkz
    kx,ky,kz=np.meshgrid(kxl,kyl,kzl,indexing='ij')
    return kx,ky,kz

def init_linmats(pars,kx,ky,kz):    
    # Initializing the linear matrices
    kapn,kapt,kapb,tau,chi,a,b,s,Dphi,Dt,Dv,Hphi,Ht,Hv,nuphi,nut,nuv = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','tau','chi','a','b','s','Dphi','Dt','Dv','Hphi','Ht','Hv','nuphi','nut','nuv']
    ]
    kpsq = kx**2 + ky**2
    kpsq[kpsq==0] = 1e-10
    kzsq = kz**2
    kzsq[kzsq==0] = 1e-10
        
    sigk = ky>0
    fac=tau+kpsq
    lm=torch.zeros(kx.shape+(3,3),dtype=torch.complex64)
    lm[:,:,:,0,0]=-1j*chi*kpsq-1j*Dt*kpsq**2-1j*nut*kzsq**2-1j*sigk*Ht/kpsq**3
    lm[:,:,:,0,1]=(5/3)*kz
    lm[:,:,:,0,2]=(kapn+kapt)*ky
    lm[:,:,:,1,0]=kz
    lm[:,:,:,1,1]=-1j*s*chi*kpsq-1j*Dv*kpsq**2-1j*nuv*kzsq**2-1j*sigk*Hv/kpsq**3
    lm[:,:,:,1,2]=kz
    lm[:,:,:,2,0]=(-kapb*ky+1j*chi*kpsq**2*b)/fac
    lm[:,:,:,2,1]=kz/fac
    lm[:,:,:,2,2]=(kapn*ky-(kapn+kapt)*ky*kpsq-kapb*ky-1j*chi*kpsq**2*a)/fac-1j*Dphi*kpsq**2-1j*nuphi*kzsq**2-1j*sigk*Hphi/kpsq**3

    return lm

def linfreq(pars, kx, ky, kz):
    lm = init_linmats(pars, torch.from_numpy(kx), torch.from_numpy(ky), torch.from_numpy(kz)).cuda()
    # print(lm.device)
    w = torch.linalg.eigvals(lm)
    iw = torch.argsort(-w.imag, -1)
    lam = torch.gather(w, -1, iw).cpu().numpy()
    # vi = torch.gather(v, -1, iw.unsqueeze(-2).expand_as(v)).cpu().numpy()
    del lm, w, iw
    torch.cuda.empty_cache()
    return lam

#%% Initialize

Npx,Npy,Npz=512,512,128
Nx,Ny,Nz=2*int(Npx/3),2*int(Npy/3),2*int(Npz/3)
Lx,Ly,Lz=32*np.pi,32*np.pi,32*np.pi
kx,ky,kz=init_kspace_grid(Nx,Ny,Nz,Lx,Ly,Lz)
kapn=0. #rho_i/L_n
kapt=1.0 #rho_i/L_T
kapb=1.0 #2*rho_i/L_B
chi=0.1
a=9.0/40.0
b=67.0/160.0
s=0.9
D0=0*1e-6
H0=0*1e-5
nu0=0*1e-6
base_pars={'kapn':kapn,
      'kapt':kapt,
      'kapb':kapb,
      'tau':1.,#Ti/Te
      'chi':chi,
      'a':a,
      'b':b,
      's':s,
      'Dphi':D0,
      'Dt':D0,
      'Dv':D0,
      'Hphi':H0,
      'Ht':H0,
      'Hv':H0,
      'nuphi':nu0,
      'nut':nu0,
      'nuv':nu0}

#%% Compute om

om=linfreq(base_pars,kx,ky,kz)
omr=om.real[:,:,:,0]
gam=om.imag[:,:,:,0]

#%% Compute max quantities

print('gammax:',np.max(gam),'1/gammax:',1/np.max(gam))
print('max index:', np.unravel_index(np.argmax(gam), gam.shape))
print('max kx:', kx[np.unravel_index(np.argmax(gam), gam.shape)])
print('max ky:', ky[np.unravel_index(np.argmax(gam), gam.shape)])
print('max kz:', kz[np.unravel_index(np.argmax(gam), gam.shape)])

ind_kzmax = np.argmax(gam, axis=2, keepdims=True)   
gam_kzmax = np.take_along_axis(gam, ind_kzmax, axis=2).squeeze(axis=2) #select gam at the max kz index
omr_kzmax = np.take_along_axis(omr, ind_kzmax, axis=2).squeeze(axis=2)
gam_kz0 = gam[:,:,0]
omr_kz0 = omr[:,:,0]

ind_kxmax = np.argmax(gam, axis=0,keepdims=True) 
gam_kxmax = np.take_along_axis(gam, ind_kxmax, axis=0).squeeze(axis=0)  #select gam at the max kx index
omr_kxmax = np.take_along_axis(omr, ind_kxmax, axis=0).squeeze(axis=0) 
gam_kx0 = gam[0,:,:]
omr_kx0 = omr[0,:,:]

#%% Plots

plt.figure()
slz=slice(None,Nz,int(Nz/8)) #9 kz points
plt.plot(ky[0,:int(Ny/4),slz],gam_kx0[:int(Ny/4),slz],'.-')
plt.plot(ky[0,:int(Ny/4),0],0*ky[0,:int(Ny/4),0],'k--')
# plt.plot(ky[0,:int(Ny/4),0],-chi*ky[0,:int(Ny/4),0]**2,'k--')
plt.legend(['$k_z='+str(l)+'$' for l in kz[0,0,slz]]+['$-\\chi k_y^2$'])
plt.xlabel('$k_y$')
plt.ylabel('$\\gamma(k_y)$')
plt.title('$\\gamma(k_{x,max}(k_y,k_z),k_y,k_{zi})$ vs $k_y$ for diff $k_z$') 
plt.savefig('data/gam_vs_ky_kzvals_itg.png',dpi=600)
plt.show()

kymax_kz= np.take_along_axis(ky[0,:int(Ny/2),:],np.argmax(gam_kx0[:int(Ny/2),:],axis=-2,keepdims=True),axis=-2).squeeze(axis=-2)
plt.figure()
plt.plot(kz[0,0,:],kymax_kz,'.-')
plt.xlabel('$k_z$')
plt.ylabel('$k_{y,max}$')
plt.title('$k_{y,max}$ vs $k_z$')
plt.savefig('data/ky_vs_kz_itg.png',dpi=600)
plt.show()
    
plt.figure()
slx=slice(None,int(Nx/2),int((Nx/2)/8)) #9 kx points
plt.plot(ky[slx,:int(Ny/2),0].T,gam_kz0[slx,:int(Ny/2)].T,'.-')
# plt.plot(ky[0,:int(Ny/2),0],-chi*ky[0,:int(Ny/2),0]**2,'k--')
plt.legend(['$k_x='+str(l)+'$' for l in kx[slx,0,0]]+['$-\\chi k_y^2$'])
plt.xlabel('$k_y$')
plt.ylabel('$\\gamma(k_y)$')
plt.title('$\\gamma(k_{xi},k_y,k_{z,max}(k_x,k_y))$ vs $k_x$ for diff $k_x$')
plt.savefig('data/gam_vs_ky_kxvals_itg.png',dpi=600)
plt.show()

kymax_kx= np.take_along_axis(ky[:int(Nx/2),:int(Ny/2),0],np.argmax(gam_kz0[:int(Nx/2),:int(Ny/2)],axis=1,keepdims=True),axis=1).squeeze(axis=1)
plt.figure()
plt.plot(kx[:int(Nx/2),0,0],kymax_kx,'.-')
plt.xlabel('$k_x$')
plt.ylabel('$k_{y,max}$')
plt.title('$k_{y,max}$ vs $k_x$')
plt.savefig('data/ky_vs_kx_itg.png',dpi=600)
plt.show()

#%% colormesh of gam and omr at kx max

ky_shifted = np.fft.fftshift(ky[0, :, :], axes=0)
kz_shifted = np.fft.fftshift(kz[0, :, :], axes=0)

plt.figure()
plt.pcolormesh(ky_shifted,kz_shifted,np.fft.fftshift(gam_kx0,axes=0),vmax=1.88,vmin=-1.88,cmap='seismic',rasterized=True)
plt.xlabel('$k_y$')
plt.ylabel('$k_z$')
plt.title('$\\gamma(k_{x,max}(k_y,k_z),k_y,k_z)$')
plt.colorbar()
plt.savefig('data/gamkykz_itg.png',dpi=600)
plt.show()

# plt.figure()
# plt.pcolormesh(ky_shifted,kz_shifted,np.fft.fftshift(omr_kx0,axes=0),vmax=0.2,vmin=-0.2,cmap='seismic',rasterized=True)
# plt.xlabel('$k_y$')
# plt.ylabel('$k_z$')
# plt.title('$\\omega_r(k_y,k_z) = \\omega_r(k_{x,max}(k_y,k_z),k_y,k_z)$')
# plt.colorbar()
# plt.savefig('data/omrkykz_itg.png',dpi=600)
# plt.show()

#%% colormesh of gam and omr at kz max

kx_shifted = np.fft.fftshift(kx[:,:,0],axes=0)
ky_shifted = np.fft.fftshift(ky[:,:,0],axes=0)

plt.figure()
plt.pcolormesh(kx_shifted[:,:int(Ny/2)], ky_shifted[:,:int(Ny/2)], np.fft.fftshift(gam_kz0[:,:int(Ny/2)], axes=0),vmax=1.88,vmin=-1.88,cmap='seismic', rasterized=True)
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.title('$\\gamma(k_x,k_y,k_{z,max}(k_x,k_y))$')
plt.colorbar()
plt.savefig('data/gamkxky_itg.png',dpi=600)
plt.show()

# plt.figure()
# plt.pcolormesh(kx_shifted[:,:int(Ny/2)], ky_shifted[:,:int(Ny/2)], np.fft.fftshift(omr_kz0[:,:int(Ny/2)], axes=0),vmax=0.2,vmin=-0.2,cmap='seismic', rasterized=True)
# plt.xlabel('$k_x$')
# plt.ylabel('$k_y$')
# plt.title('$\\omega_r(k_x,k_y) = \\omega_r(k_x,k_y,k_{z,max}(k_x,k_y))$')
# plt.colorbar()
# plt.savefig('data/omrkxky_itg.png',dpi=600)
# plt.show()

#%% colormesh of kzmax

# plt.figure() 
# plt.pcolormesh(kx_shifted[:,:int(Ny/2)], ky_shifted[:,:int(Ny/2)], np.fft.fftshift(np.take_along_axis(kz[:,:,:], ind_kzmax, 2).squeeze(axis=2)[:,:int(Ny/2)], axes=0),vmax=.8,vmin=-.8,cmap='seismic')
# plt.xlabel('$k_x$')
# plt.ylabel('$k_y$')
# plt.title('$k_{z,max}(k_x,k_y)$')
# plt.colorbar()
# plt.savefig('data/kzmax_itg.png',dpi=600)
# plt.show()

