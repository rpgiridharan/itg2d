#%% Import modules
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch 

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['axes.linewidth'] = 3  
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.width'] = 1.5 
plt.rcParams['ytick.minor.width'] = 1.5 
plt.rcParams['savefig.dpi'] = 100
plt.rcParams.update({
    "font.size": 22,          # default text
    "axes.titlesize": 30,     # figure title
    "axes.labelsize": 26,     # x/y labels
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22
})

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
    kapn,kapt,kapb,tau,D,H = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','tau','D','H']
    ]
    kpsq = kx**2 + ky**2
    kpsq = torch.where(kpsq==0, 1e-10, kpsq)
        
    sigk = ky>0
    fac=tau*sigk+kpsq
    lm=torch.zeros(kx.shape+(2,2),dtype=torch.complex64)
    lm[:,:,0,0]=-1j*sigk*D*kpsq-1j*sigk*H/kpsq**2
    lm[:,:,0,1]=(kapn+kapt)*ky
    lm[:,:,1,0]=-kapb*ky/fac
    lm[:,:,1,1]=(kapn*ky-(kapn+kapt)*ky*kpsq)/fac-1j*sigk*D*kpsq-1j*sigk*H/kpsq**2

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

def one_over(x):
    out = np.zeros_like(x)
    return np.divide(1.0, x, out=out, where=x != 0)

#%% Initialize

Npx,Npy=4096,4096
Nx,Ny=2*int(Npx/3),2*int(Npy/3)
# Lx,Ly=32*np.pi,32*np.pi #sim for 512x512
Lx,Ly=256*np.pi,256*np.pi 
kx,ky=init_kspace_grid(Nx,Ny,Lx,Ly)
kapt=0.4 #rho_i/L_T >0.2
kapn=0.2 #rho_i/L_n
kapb=0.02 #2*rho_i/L_B
D=1e-3 #0.1
H= 0*1e-06 # 4.2581e-06 for kapt=0.5
base_pars={'kapn':kapn,
      'kapt':kapt,
      'kapb':kapb,
      'tau':1.,#Ti/Te
      'D':D,
      'H':H}

#%% Compute om

om=linfreq(base_pars,kx,ky)
omr=om.real[:,:,0]
gam=om.imag[:,:,0]
Dturb=gam*one_over(kx**2+ky**2)
gammax=np.max(gam)
Dturbmax=np.max(Dturb)

#%% Compute quantities

print('gammax:',gammax,'1/gammax:',1/gammax)
print('max index:', np.unravel_index(np.argmax(gam[:,:]), gam.shape))
print('max kx:', kx[np.unravel_index(np.argmax(gam[:,:]), gam.shape)])
print('max ky:', ky[np.unravel_index(np.argmax(gam[:,:]), gam.shape)])

ind_kxmax = np.argmax(gam, axis=0, keepdims=True) 
gam_kxmax = np.take_along_axis(gam, ind_kxmax, axis=0).squeeze(axis=0)  #select gam at the max kx index
omr_kxmax = np.take_along_axis(omr, ind_kxmax, axis=0).squeeze(axis=0) 
kxmax_ky= np.take_along_axis(kx, ind_kxmax, axis=0).squeeze(axis=0) #select kx at the max kx index
gam_kx0 = gam[0,:]
omr_kx0 = omr[0,:]

ind_kxmax_Dturb = np.argmax(Dturb, axis=0, keepdims=True)
Dturb_kxmax = np.take_along_axis(Dturb, ind_kxmax_Dturb, axis=0).squeeze(axis=0)
kxmax_ky_Dturb = np.take_along_axis(kx, ind_kxmax_Dturb, axis=0).squeeze(axis=0)
Dturb_kx0 = Dturb[0,:]

print('Dturbmax:',Dturbmax,'1/Dturbmax:',1/Dturbmax)
print('max index:', np.unravel_index(np.argmax(Dturb[:,:]), Dturb.shape))
print('max kx:', kx[np.unravel_index(np.argmax(Dturb[:,:]), Dturb.shape)])
print('max ky:', ky[np.unravel_index(np.argmax(Dturb[:,:]), Dturb.shape)])

kx_shifted = np.fft.fftshift(kx, axes=0)
ky_shifted = np.fft.fftshift(ky, axes=0)
gam_shifted = np.fft.fftshift(gam, axes=0)

#%% Plots

plt.figure(figsize=(9.71,6))
plt.plot(ky[0,:int(Ny/8)].T,gam_kxmax[:int(Ny/8)].T,'.-',label='$k_x= \\arg\\max_{k_x} \\left(\\gamma\\right)$')
plt.plot(ky[0,:int(Ny/8)].T,gam_kx0[:int(Ny/8)].T,'.-',label='$k_x=0$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.plot(ky[0,:int(Ny/8)],-D*ky[0,:int(Ny/8)]**2,'k--',label='$-Dk_y^2$')
plt.legend()
plt.grid(which='major', linestyle='--', linewidth=0.5)
plt.xlabel('$k_y$')
plt.ylabel('$\\gamma(k_y)$')
plt.title('$\\gamma(k_y)$ vs $k_y$')
plt.tight_layout()
plt.savefig(f'data_linear/gam_vs_ky_kapt_{str(kapt).replace(".", "_")}_itg2d.pdf',dpi=100)
plt.show()

plt.figure(figsize=(9.71,6))
# slx=slice(None,int(Nx/32),1) 
slx=slice(None,int(Nx/8),int((Nx/8)/5)) #7 kx points
plt.plot(ky[slx,:int(Ny/4)].T,gam[slx,:int(Ny/4)].T,'.-')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.plot(ky[0,:int(Ny/4)],-D*ky[0,:int(Ny/4)]**2,'k--')
plt.grid(which='major', linestyle='--', linewidth=0.5)
plt.legend(['$k_x='+str(l)+'$' for l in kx[slx,0]]+['$-Dk_y^2$'])
plt.xlabel('$k_y$')
plt.ylabel('$\\gamma(k_y)$')
plt.title('$\\gamma(k_{xi},k_y)$ vs $k_x$ for diff $k_x$')
plt.tight_layout()
plt.savefig(f'data_linear/gam_vs_ky_kxvals_kapt_{str(kapt).replace(".", "_")}_itg2d.pdf',dpi=100)
plt.show()

# plt.figure()
# plt.plot(ky[0,:int(Ny/4)],kxmax_ky[:int(Ny/4)],'.-')
# plt.xlabel('$k_y$')
# plt.ylabel('$k_{x,max}$')
# plt.title('$k_{x,max}$ vs $k_y$')
# plt.tight_layout()
# plt.savefig(f'data_linear/ky_vs_kx_kapt_{str(kapt).replace(".", "_")}_itg2d.pdf',dpi=100)
# plt.show()

# kymax_kx= np.take_along_axis(ky[:int(Nx/4),:],np.argmax(gam[:int(Nx/4),:],axis=1,keepdims=True),axis=1).squeeze(axis=1)
# plt.figure()
# plt.plot(kx[:int(Nx/4),0],kymax_kx,'.-')
# plt.xlabel('$k_x$')
# plt.ylabel('$k_{y,max}$')
# plt.title('$k_{y,max}$ vs $k_x$')
# plt.tight_layout()
# plt.savefig(f'data_linear/ky_vs_kx_kapt_{str(kapt).replace(".", "_")}_itg2d.pdf',dpi=100)
# plt.show()

plt.figure(figsize=(9.71,6))
plt.plot(ky[0,:int(Ny/8)].T,Dturb_kxmax[:int(Ny/8)].T,'.-',label='$k_x= \\arg\\max_{k_x} \\left(\\frac{\\gamma}{k_\\perp^2}\\right)$')
plt.plot(ky[0,:int(Ny/8)].T,Dturb_kx0[:int(Ny/8)].T,'.-',label='$k_x=0$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.plot(ky[0,:int(Ny/8)],-D*np.ones_like(ky[0,:int(Ny/8)]),'k--',label='$-D$')
plt.legend()
plt.grid(which='major', linestyle='--', linewidth=0.5)
plt.xlabel('$k_y$')
plt.ylabel('$\\left(\\frac{\\gamma}{k_y^2}\\right)$')
plt.title('$\\left(\\frac{\\gamma}{k_y^2}\\right)$ vs $k_y$')
plt.tight_layout()
plt.savefig(f'data_linear/chi_vs_ky_kapt_{str(kapt).replace(".", "_")}_itg2d.pdf',dpi=100)
plt.show()

#%% colormesh of gam and omr

# Now slice the central half
kx_central = kx_shifted[int(3*Nx/8):int(5*Nx/8), :int(Ny/8)]
ky_central = ky_shifted[int(3*Nx/8):int(5*Nx/8), :int(Ny/8)]
gam_central = gam_shifted[int(3*Nx/8):int(5*Nx/8), :int(Ny/8)]

plt.figure(figsize=(9.71,6))
plt.pcolormesh(kx_central, ky_central, gam_central,vmax=gammax,vmin=-gammax,cmap='seismic', rasterized=True)
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.title('$\\gamma(k_x,k_y)$')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'data_linear/gamkxky_kapt_{str(kapt).replace(".", "_")}_itg2d.pdf',dpi=100)
plt.show()

# plt.figure()
# plt.pcolormesh(kx_shifted, ky_shifted, np.fft.fftshift(omr, axes=0),vmax=0.2,vmin=-0.2,cmap='seismic', rasterized=True)
# plt.xlabel('$k_x$')
# plt.ylabel('$k_y$')
# plt.title('$\\omega_r(k_x,k_y) = \\omega_r(k_x,k_y)$')
# plt.colorbar()
# plt.tight_layout()
# plt.savefig(f'data_linear/omrkxky_kapt_{str(kapt).replace(".", "_")}_itg2d.pdf',dpi=100)
# plt.show()
