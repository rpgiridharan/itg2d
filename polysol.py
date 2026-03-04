#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:34:23 2026

@author: ogurcan
"""

import numpy as np
import matplotlib.pylab as plt

# k = np.array([0,1])
# p = np.array([0.2,0.2])
# q = -k-p

k = np.array([0,1])
p = np.array([0.3,1.4])
q = -k-p

# k = np.array([0,1])
# p = np.array([0.1,-0.7])
# q = -k-p

ksq,psq,qsq=np.sum(k**2),np.sum(p**2),np.sum(q**2)
thk = np.arange(-np.pi,np.pi,np.pi/64)

#M,L,N=0,1,0
M,L,N=1,1,1
#M,L,N=1,0,1
#M,L,N=0,1,1
#M,L,N=1,1,0

S = (p[0]*q[1]-q[0]*p[1])
Mkpq = M*S
Nkpq = -N*S/(1+ksq)
Lkpq = L*S*(qsq-psq)/(1+ksq)

Mpqk = M*S
Npqk = -N*S/(1+psq)
Lpqk = L*S*(ksq-qsq)/(1+psq)

Mqkp = M*S
Nqkp = -N*S/(1+qsq)
Lqkp = L*S*(psq-ksq)/(1+qsq)

pdotq=(q[0]*p[0]+q[1]*p[1])
pdotk=(k[0]*p[0]+k[1]*p[1])
qdotk=(k[0]*q[0]+k[1]*q[1])

A=Mpqk*(Mqkp-Nqkp*qdotk*np.exp(-1j*thk))
B=Lpqk*(Nqkp*pdotq*np.exp(1j*thk)-Lqkp)+Npqk*(-pdotq*Lqkp*np.exp(-1j*thk)+pdotk*Mqkp*np.exp(1j*thk)+Nqkp*(pdotq)**2)
C=Mpqk*(Lqkp*np.exp(1j*thk)+Mqkp*np.exp(1j*thk)-Nqkp*pdotq)
D=Lpqk*Nqkp*qdotk+Npqk*(pdotq*qdotk*Nqkp*np.exp(-1j*thk)-pdotk*Mqkp)

res=[]
for l in range(len(thk)):
#    pol = [1,A[l]+B[l],A[l]*B[l]-C[l]*D[l]]
    pol = [1,0,A[l]+B[l],0,A[l]*B[l]-C[l]*D[l]]
    r=np.roots(pol)
    res.append(r[np.flipud(np.argsort(r.real))])
gam=np.array(res)
plt.plot(thk,gam.real)