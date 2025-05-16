import numpy as np
import cupy as cp

def ky_max(ky, kapt):
    # Convert CuPy array to NumPy if needed
    if isinstance(ky, cp.ndarray):
        ky = ky.get()
        
    kpsq = ky**2
    p2 = (1+kpsq)
    p1 = -1j*ky*(1+kapt*kpsq)
    p0 = -kapt*kpsq
    
    # Use NumPy operations instead of CuPy
    coeffs = np.stack((p2, p1, p0), axis=-1)
    roots = np.array([np.roots(p) for p in coeffs])
    gamk = np.real(roots)
    gamk_max = np.take_along_axis(gamk, np.argmax(gamk, axis=-1, keepdims=True), axis=-1).squeeze(axis=-1)
    return ky[np.argmax(gamk_max)]

def gam_max(ky, kapt):
    # Convert CuPy array to NumPy if needed
    if isinstance(ky, cp.ndarray):
        ky = ky.get()
        
    kpsq = ky**2
    p2 = (1+kpsq)
    p1 = -1j*ky*(1+kapt*kpsq)
    p0 = -kapt*kpsq
    
    # Use NumPy operations instead of CuPy
    coeffs = np.stack((p2, p1, p0), axis=-1)
    roots = np.array([np.roots(p) for p in coeffs])
    gamk = np.real(roots)
    gamk_max = np.take_along_axis(gamk, np.argmax(gamk, axis=-1, keepdims=True), axis=-1).squeeze(axis=-1)
    return np.max(gamk_max)