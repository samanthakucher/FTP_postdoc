import os
import h5py
import numpy as np
import numpy.ma as ma
import yaml
import skimage.io as sio
from tqdm import tqdm
from scipy.signal import find_peaks
from natsort import natsorted
import glob
import time
from scipy.special import erf


import xarray as xr
import dask
from dask.distributed import Client, LocalCluster
from dask import delayed
import dask.array as da

from parallelized import perform_fft2_per_frame_and_keep_solid_lines


from matplotlib import cm
import matplotlib.pyplot as plt
plt.ion()

from matplotlib import rc
import matplotlib as mpl
fs = 15
rc('legend', fontsize=12)
rc('axes',   labelsize=fs)
rc('xtick',  labelsize=fs)
rc('ytick',  labelsize=fs)
rc('lines',  markersize=7)
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']+ mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = "stix"

import multiprocessing
# multiprocessing.set_start_method("fork")
from multiprocessing import Pool, cpu_count
from itertools import repeat
import gc

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

cores = cpu_count()
cores = 10


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


#if __name__ == "__main__":

def perform_fft_and_save_planes_solid(measurements_path, folder_name, periodicity=24, N_vertical_slices=cores, plot_theorical_disp_relation=True):
    """
    Performs 2D FFT in each frame and saves the planes corresponding to the solid structure.

    Input:
    - measurements_path : string
    directory where all the measurements are (including gray and reference)
    - folder_name : string
    name of the folder that contains the height fields
    - periodicity : int, optional (default=24)
    Periodicity of the solid bottom, in mm
    - N_vertical_slices : int, optional (default=cores)
    Number of parallel workers

    Output:
    Saves the dispersion relation in .npy format.

    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    ftp_nc_path = measurements_path + folder_name + '/'

    logger.info('---------- FOURIER TRANSFORM ----------')

    # logger.info('Reading ftp_processing_parameters')
    parameter_file = measurements_path + 'processing_parameters.yaml'
    ftp_proc_parameters = yaml.safe_load(open(parameter_file))
    fps         = ftp_proc_parameters['MEASUREMENT']['fps']
    pixel_size  = ftp_proc_parameters['MEASUREMENT']['pixel_size']

    height_fields_files = natsorted(glob.glob(os.path.join(ftp_nc_path, '*.nc')), key=lambda y: y.lower())

    h0 = xr.open_dataarray(height_fields_files[0]) 
    Slin, Scol = h0.shape
    N = len(height_fields_files)


    kx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Slin, pixel_size))

    # Bragg wavenumber
    KB = 2*np.pi/periodicity
    idx_kxB = find_nearest(kx, KB)
    kxB = kx[idx_kxB]

    plane_GX = np.zeros((Slin, N), dtype='c16')
    plane_XM = np.zeros((Slin, N), dtype='c16')
    plane_MG = np.zeros((Slin, N), dtype='c16')


    t1 = time.time()

    # FFT 2D
    radius = Slin//2
    xc, yc = Slin//2, Slin//2
    xx, yy = np.arange(Slin), np.arange(Slin)
    (xi, yi) = np.meshgrid(xx,yy)
    error_function = -erf((xi-xc)**2+(yi-yc)**2-radius**2)+1

    # Paralellized
    iteration_time = []
    for j in tqdm(range(int(N/N_vertical_slices))):
        non_iterable_args = height_fields_files, error_function, Scol, idx_kxB 

        ti = time.time() 
        p = Pool(cores)
        lines_ft2_p = p.starmap(perform_fft2_per_frame_and_keep_solid_lines, zip(np.arange(N_vertical_slices*j, N_vertical_slices*(j+1)), repeat(non_iterable_args)))
        p.close()

        line_GX, line_XM, line_MG = np.dstack(lines_ft2_p)
        plane_GX[:,(N_vertical_slices*j):(N_vertical_slices*(j+1))] = line_GX
        plane_XM[:,(N_vertical_slices*j):(N_vertical_slices*(j+1))] = line_XM
        plane_MG[:,(N_vertical_slices*j):(N_vertical_slices*(j+1))] = line_MG

        gc.collect()

        tf = time.time() - ti
        iteration_time.append(tf)


    # plane_GX = np.abs(np.fft.fftshift(np.fft.fft(plane_GX, axis=1), axes=1))
    # plane_XM = np.abs(np.fft.fftshift(np.fft.fft(plane_XM, axis=1), axes=1))
    # plane_MG = np.abs(np.fft.fftshift(np.fft.fft(plane_MG, axis=1), axes=1))

    plane_GX = np.fft.fftshift(np.fft.fft(plane_GX, axis=1), axes=1)
    plane_XM = np.fft.fftshift(np.fft.fft(plane_XM, axis=1), axes=1)
    plane_MG = np.fft.fftshift(np.fft.fft(plane_MG, axis=1), axes=1)

    fmax = 20
    frequencies = np.fft.fftfreq(N, 1/fps)
    frequencies  = np.fft.fftshift(frequencies)
    idx_fmax = find_nearest(frequencies, fmax)

    plane_GX_BZ = plane_GX[Slin//2:idx_kxB,N//2:idx_fmax]
    plane_XM_BZ = plane_XM[Slin//2:idx_kxB, N//2:idx_fmax]
    plane_MG_BZ = np.fliplr(np.transpose(plane_MG[Slin//2:idx_kxB, N//2:idx_fmax]))

    planes_total = np.concatenate((np.transpose(plane_GX_BZ), np.transpose(plane_XM_BZ), plane_MG_BZ), axis=-1)

    np.save(ftp_nc_path + 'plane_GX.npy', plane_GX)
    np.save(ftp_nc_path + 'plane_XM.npy', plane_XM)
    np.save(ftp_nc_path + 'plane_MG.npy', plane_MG)
    np.save(ftp_nc_path+ 'reldisp.npy', planes_total)

    t2 = time.time()

    dt = t2 - t1
    #print(dt)
    print('Time for Fourier Transform = '+str(round(dt,2)) + 's')

    plt.figure()
    plt.plot(iteration_time,'.-')
    plt.xlabel('Number of iteration')
    plt.ylabel('Iteration time [s]')
    plt.title('Fourier Transform')
    plt.grid()
    plt.show()


    plt.figure()
    im = plt.imshow(np.log(np.abs((planes_total))), cmap='turbo', aspect='auto', origin='lower', extent=[0, 3*KB, 0, frequencies[idx_fmax]])
    plt.clim(9, 13)
    if plot_theorical_disp_relation == True:
        D = np.load(measurements_path + 'reldisp_teorica.npy')
        kxD = np.linspace(0, 3*KB, D.shape[1])
        fD = np.linspace(0,fmax, D.shape[0])
        plt.contour(kxD, fD, np.flipud(D) ,levels=[0],colors="w")
    plt.vlines(KB, 0, 20, 'w', 'dashed')
    plt.vlines(2*KB, 0, 20, 'w', 'dashed')
    plt.xlabel('$K$')
    plt.ylabel('$f$ [Hz]')
    plt.ylim([0,fmax])
    #plt.title('Dispersion relation')
    plt.title(folder_name)
    plt.xticks([0, KB, 2*KB, 3*KB], [r'$\Gamma$', 'X', 'M', '$\Gamma$'])
    plt.colorbar(im)
    plt.show()
    plt.savefig(ftp_nc_path + 'reldisp.png')

    return None



