import numpy as np
import numpy.ma as ma
from scipy import signal
from unwrap import unwrap
#from FTP_postdoc.fringe_extrapolation import gerchberg2d
#from fringe_extrapolation import gerchberg2d

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

def calculate_phase_diff_map_1D(dY, dY0, th, ns, mask_for_unwrapping=None):
    """
    # % Basic FTP treatment.
    # % This function takes a deformed and a reference image and calculates the phase difference map between the two.
    # %
    # % INPUTS:
    # % dY	= deformed image
    # % dY0	= reference image
    # % ns	= size of gaussian filter
    # %
    # % OUTPUT:
    # % dphase 	= phase difference map between images
    """

    nx, ny = np.shape(dY)
    # phase0 = np.zeros([nx,ny])
    # phase  = np.zeros([nx,ny])
    phase0 = np.zeros_like(dY0)
    phase  = np.zeros_like(dY)


    #for lin in range (0,nx):
    for lin in range(nx):
        fY0=np.fft.fft(dY0[lin,:])
        fY=np.fft.fft(dY[lin,:])

        dfy=1./ny
        fy=np.arange(dfy,1,dfy)

        imax=np.argmax(np.abs(fY0[9:int(np.floor(ny/2))]))
        ifmax=imax+9

        HW=np.round(ifmax*th)
        #W=2*HW
        W=2*HW+1
        win=signal.windows.tukey(int(W),ns)


        #gaussfilt1D= np.zeros([1,nx])
        gaussfilt1D= np.zeros(ny)
        #gaussfilt1D[int(ifmax-HW-1):int(ifmax-HW+W-1)]=win
        gaussfilt1D[int(ifmax-HW):int(ifmax-HW+W)]=win

        # Multiplication by the filter
        Nfy0 = fY0*gaussfilt1D
        Nfy = fY*gaussfilt1D

        # Inverse Fourier transform of both images
        Ny0=np.fft.ifft(Nfy0)
        Ny=np.fft.ifft(Nfy)
        
        phase0[lin,:] = np.unwrap(np.angle(Ny0))
        phase[lin,:]  = np.unwrap(np.angle(Ny))

        # phase0[lin,:] = np.angle(Ny0)
        # phase[lin,:]  = np.angle(Ny)
    
    # 2D-unwrapping is available with masks (as an option), using 'unwrap' library
    # unwrap allows for the use of wrapped_arrays, according to the docs:
    # "[...] in this case masked entries are ignored during the phase unwrapping process. This is useful if the wrapped phase data has holes or contains invalid entries. [...]"


    mphase0 = unwrap(phase0)
    mphase = unwrap(phase)
    if mask_for_unwrapping is None:
        mphase0 = unwrap(phase0)
        mphase = unwrap(phase)
    else:
        mask_for_unwrapping = np.array(mask_for_unwrapping)
        mphase0 = ma.masked_array(phase0, mask=mask_for_unwrapping)
        mphase  = ma.masked_array(phase,  mask=mask_for_unwrapping)
        mphase0 = unwrap(mphase0)
        mphase = unwrap(mphase)
    
    # Definition of the phase difference map
    dphase = (mphase-mphase0);
    # dphase = unwrap(dphase)
    #dphase = dphase - np.min(dphase) - np.pi/2 

    # dphase = (dphase + np.pi) % (2 * np.pi) - np.pi
    #dphase = np.arctan2(np.sin(dphase), np.cos(dphase))


    # if mask_for_unwrapping is None:
    #     pass
    # else:
    #     division  = np.mean(ma.masked_array(dphase, mask=mask_for_unwrapping)) // (np.pi/2)
    #     #if np.abs(division)>=1:
    #     if np.abs(division)>1:
    #         dphase = dphase - division*(np.pi/2)
    # division  = np.mean(dphase) // (np.pi/2)
    # if np.abs(division)>=1:
    #     dphase = dphase - division*(np.pi/2)

    return dphase



def height_map_from_phase_map(dphase, L, D, p):
    """
    Converts a phase difference map to a height map using the phase to height
    relation.
    
    INPUTS:
         dphase    = phase difference map (already unwrapped)
         L         = distance between the reference surface and the plane of the entrance  pupils
         D         = distance between centers of entrance pupils
         p         = wavelength of the projected pattern (onto the reference surface)
         spp       = physical size of the projected pixel (as seen onto the reference  surface)
        
         OUTPUT:
            h         = height map of the surface under study
"""
    return -L*dphase/(2*np.pi/p*D-dphase)

