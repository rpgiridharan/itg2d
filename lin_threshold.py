#%% import libraries
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 3  
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.width'] = 1.5 
plt.rcParams['ytick.minor.width'] = 1.5 

# %% Initialize parameters and arrays

dk=0.01
k = np.arange(0,5.0,dk) 
# kapb=np.arange(1/90,1/62.5,1e-4)/0.22
kapb=np.arange(1e-4,1,1e-4)

K,Kapb=np.meshgrid(k,kapb,indexing='ij')

thresh_min_candidate=np.zeros_like(K)
thresh_max_candidate=np.zeros_like(K)

thresh_min_candidate[0,:]=np.maximum(0.0,(1+Kapb[0,:]**2-6*Kapb[0,:])/(4*Kapb[0,:]))
thresh_max_candidate[0,:]=1/np.float64('inf')

sD=2*np.sqrt(Kapb)*np.sqrt(Kapb*K**2+Kapb+K**4+K**2)
common=Kapb*K**2+2*Kapb-K**4+K**2
thresh_min_candidate[1:,:] = np.maximum(0.0, (-sD + common)[1:,:]/K[1:,:]**4)
thresh_max_candidate[1:,:] = (sD+common)[1:,:]/K[1:,:]**4

thresh_min = np.min(thresh_min_candidate,axis=0)
thresh_max = np.max(thresh_max_candidate,axis=0)

k_min = np.argmin(thresh_min_candidate,axis=0)
k_max = np.argmax(thresh_max_candidate,axis=0)

# %% Plots
plt.figure()
plt.plot(kapb, thresh_min, label='min')
plt.plot(kapb, thresh_max, label='max')
plt.xlabel('$\\bar{\\kappa}_B$')
plt.ylabel(f'$\\bar{{\\kappa}}_{{T,lin}}$')
plt.title(f'Domain of $\\bar{{\\kappa}}_T$ for $\\gamma>0$ vs $\\bar{{\\kappa}}_B$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('data_linear/lin_thresh_itg2d.png', dpi=600)
plt.show()

plt.figure()
plt.plot(kapb, k[k_min], label='k at min thresh')
plt.plot(kapb, k[k_max], label='k at max thresh')
plt.axhline(dk, color='k', linestyle=':', label='$k = dk$')
plt.xlabel('$\\bar{\\kappa}_B$')
plt.ylabel('$k$')
plt.title(f'k at min and max $\\bar{{\\kappa}}_T$ s.t. $\\gamma>0$ vs $\\bar{{\\kappa}}_B$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('data_linear/k_lin_thresh_itg2d.png', dpi=600)
plt.show()