import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy.signal import hann

def gerchberg2d(interferogram, mask_where_fringes_are, N_iter_max):
    """
    Extrapolates fringe pattern beyond mask, following Gerchberg algorithm.
    """

    ref = interferogram
    refh = interferogram*mask_where_fringes_are
    interf = mask_where_fringes_are

    ft_ref  = np.fft.rfft2(ref)
    ft_refh = np.fft.rfft2(refh)

    S = ref.shape
    S = S[0]

    # k0x and R_in_k_space determination by gaussian fir
    y = (np.abs(ft_refh[0,:]))
    y = y/np.max(y)
    x = np.linspace(0, (len(y)-1), len(y))
    maxInd = argrelextrema(y, np.greater)
    x, y = x[maxInd], y[maxInd]
    n = len(x)
    w = hann(n)
    y = y*w
    index_mean = np.argwhere(y==np.max(y))[0,0]
    mean =  maxInd[0][index_mean]
    sigma = np.sum(y*(x-mean)**2)/n
    try:
        popt, pcov = curve_fit(gaus, x, y, p0 = [y[index_mean], mean, sigma],maxfev=1100)

        # popt, pcov = curve_fit(gaus, x, y, p0 = [1, mean, sigma],maxfev=1100)
    except:
        popt, pcov = curve_fit(gaus, x, y,maxfev=1100)
    '''
    try:
        popt, pcov = curve_fit(gaus, x, y)
    except RuntimeError:
        try:
            popt, pcov = curve_fit(gaus, x, y, p0 = [1, mean, sigma])
        except:
            try:
                popt, pcov = curve_fit(gaus, x[1:], y[1:], p0 = [1, mean, sigma])
            except:
                popt, pcov = curve_fit(gaus, x[1:], y[1:])
        #popt, pcov = curve_fit(gaus, x[5:], y[5:])#, p0 = [1, mean, sigma])
        #popt, pcov = curve_fit(gaus, x, y, p0 = [1, mean, sigma])
    #except RuntimeError:
        #popt, pcov = curve_fit(gaus, x[5:], y[5:])#, p0 = [1, mean, sigma])
    #except RuntimeError:
        #popt = [mean, sigma]
    '''

    k0x, k0y = popt[1], 0
    R_in_k_space = popt[2]#*2.5

    kx, ky = np.meshgrid(range(int(S/2+1)), range(S))

    # lugar_a_conservar son dos cuartos de circulo
    # centrados en 0,0 y en 0,1024
    cuarto_superior = ( (kx-k0x)**2 + (ky-(S-k0y))**2 <= R_in_k_space**2 )
    cuarto_inferior = ( (kx-k0x)**2 + (ky-(0-k0y))**2 <= R_in_k_space**2 )
    lugar_a_conservar = cuarto_inferior + cuarto_superior
    lugar_a_anular = 1-lugar_a_conservar

    # non-fancy indexing es mejor
    lugar_a_anular = lugar_a_anular.nonzero()
    interf = interf.nonzero()

    En = np.zeros(N_iter_max+1)

    ii = 0
    while ii<=N_iter_max:
        # print(ii)
        ft_refh[lugar_a_anular] = 0
        refhc = np.fft.irfft2(ft_refh)
        refhc[interf] = refh[interf]
        ft_refh = np.fft.rfft2(refhc)
        En[ii] = np.sum(np.abs(ft_refh))
        if ii > 0 and En[ii-1] < En[ii]:
            break
        ii += 1
    En = En[0:ii]

    refhc = np.real(refhc)
    refhc[interf] = ref[interf]

    return refhc


def gaus(x, a, x0, sigma):
    """
    Internal function, used to fit a gaussian.
    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
