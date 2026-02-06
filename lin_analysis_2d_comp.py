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
    "legend.fontsize": 18
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
base_pars={'kapn':kapn,
    'kapt':kapt,
    'kapb':kapb,
    'tau':1.,#Ti/Te
    'D':0.0,
    'H':0.0}

#%% Compute om

cases = [
    {'D': 0.0, 'H': 0.0, 'label': '$D=0, H=0$'},
    {'D': 1e-2, 'H': 0.0, 'label': '$D=10^{-2}, H=0$'},
    {'D': 0.0, 'H': 2e-5, 'label': '$D=0, H=2\\times10^{-5}$'},
    {'D': 1e-2, 'H': 2e-5, 'label': '$D=10^{-2}, H=2\\times10^{-5}$'},
]

results = []
for case in cases:
    base_pars.update({'D': case['D'], 'H': case['H']})
    om = linfreq(base_pars, kx, ky)
    omr = om.real[:, :, 0]
    gam = om.imag[:, :, 0]
    Dturb = gam * one_over(kx**2 + ky**2)
    results.append({
        'case': case,
        'omr': omr,
        'gam': gam,
        'Dturb': Dturb,
        'gammax': np.max(gam),
        'Dturbmax': np.max(Dturb)
    })

#%% Compute quantities

for item in results:
    gam = item['gam']
    case = item['case']
    print('case:', case['label'])
    print('gammax:', item['gammax'], '1/gammax:', 1/item['gammax'])
    print('max index:', np.unravel_index(np.argmax(gam[:,:]), gam.shape))
    print('max kx:', kx[np.unravel_index(np.argmax(gam[:,:]), gam.shape)])
    print('max ky:', ky[np.unravel_index(np.argmax(gam[:,:]), gam.shape)])

    ind_kxmax = np.argmax(gam, axis=0, keepdims=True)
    item['gam_kxmax'] = np.take_along_axis(gam, ind_kxmax, axis=0).squeeze(axis=0)
    item['omr_kxmax'] = np.take_along_axis(item['omr'], ind_kxmax, axis=0).squeeze(axis=0)
    item['kxmax_ky'] = np.take_along_axis(kx, ind_kxmax, axis=0).squeeze(axis=0)
    item['gam_kx0'] = gam[0,:]
    item['omr_kx0'] = item['omr'][0,:]

    ind_kxmax_Dturb = np.argmax(item['Dturb'], axis=0, keepdims=True)
    item['Dturb_kxmax'] = np.take_along_axis(item['Dturb'], ind_kxmax_Dturb, axis=0).squeeze(axis=0)
    item['kxmax_ky_Dturb'] = np.take_along_axis(kx, ind_kxmax_Dturb, axis=0).squeeze(axis=0)
    item['Dturb_kx0'] = item['Dturb'][0,:]

#%% Plots

slny8 = slice(1, int(Ny/8))
slny16 = slice(1, int(Ny/16))
plt.figure(figsize=(16,9))
for item in results:
    plt.plot(ky[0,slny16].T, item['gam_kxmax'][slny16].T, '.-', label=item['case']['label'])
plt.plot(ky[0,slny16], -0.01*ky[0,slny16]**2, 'k--', label=f'$-Dk_y^2$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.legend()
plt.grid(which='major', linestyle='--', linewidth=0.5)
plt.xlabel('$k_y$')
plt.ylabel('$\\gamma(k_y)$')
plt.title('$\\gamma(k_y)$ vs $k_y$')
# plt.ylim(-4*results[0]['gammax'], 2*results[0]['gammax'])
plt.tight_layout()
plt.savefig(f'data_linear/gam_vs_ky_kapt_{str(kapt).replace(".", "_")}_itg2d_comp.pdf',dpi=100)
plt.show()

plt.figure(figsize=(16,9))
for item in results:
    plt.plot(ky[0,slny16].T, item['Dturb_kxmax'][slny16].T, '.-', label=item['case']['label'])
plt.plot(ky[0,slny16], -0.01*np.ones_like(ky[0,slny16]), 'k--', label=f'$-D$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.legend()
plt.grid(which='major', linestyle='--', linewidth=0.5)
plt.xlabel('$k_y$')
plt.ylabel('$\\left(\\frac{\\gamma}{k_y^2}\\right)$')
plt.title('$\\left(\\frac{\\gamma}{k_y^2}\\right)$ vs $k_y$')
plt.ylim(-0.1*results[2]['Dturbmax'], 1.5*results[2]['Dturbmax'])
plt.tight_layout()
plt.savefig(f'data_linear/Dturb_vs_ky_kapt_{str(kapt).replace(".", "_")}_itg2d_comp.pdf',dpi=100)
plt.show()
