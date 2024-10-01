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

import xarray as xr
import dask
from dask.distributed import Client, LocalCluster
from dask import delayed
import dask.array as da

from ftp import calculate_phase_diff_map_1D, height_map_from_phase_map
from parallelized import analyze_with_ftp_and_save_to_nc


from matplotlib import cm
import matplotlib.pyplot as plt
plt.ion()

import multiprocessing
#multiprocessing.set_start_method("fork")
from multiprocessing import Pool, cpu_count
from itertools import repeat
import gc

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

cores = cpu_count()
cores = 10




# measurements_path = '/Volumes/T7 Shield/2024-07-16/monochromatic/'  
# folder_name = 'a7.5-f7'
# ftp_im_path = measurements_path + 'def-' + folder_name + '/'

# measurements_path = '/Volumes/T7 Shield/2024-07-16/sweep/'  
# folder_name = 'a6.5-f1-20-t60'
# ftp_im_path = measurements_path + 'sweep-' + folder_name + '/'
# 
# ftp_nc_path = measurements_path + folder_name + '/'
# 
# masked = True



# if __name__ == "__main__":

def process_images_by_ftp(measurements_path, folder_name, ftp_im_path, masked=True, N_vertical_slices=cores):
    """
    Performs FTP analysis from .tiff files and saves the results in netCDF format in a new folder.

    Input:
    - measurements_path : string
    directory where all the measurements are (including gray and reference)
    - folder_name : string
    name of the folder (to be created) that will contain the analyzed data
    - ftp_im_path : string
    directory where the deformed images are
    - masked : bool, optional (default=True)
    Indicates if a mask is needed to analyze the data
    - N_vertical_slices : int, optional (default=cores)
    Number of parallel workers

    Output:
    Saves the height fields in the new folder.

    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    ftp_nc_path = measurements_path + folder_name + '/'
    if not os.path.exists(ftp_nc_path):
        logger.info('Create new directory for netCDF files')
        os.makedirs(ftp_nc_path)




    logger.info('---------- FTP ANALYSIS ----------')
    parameter_file = measurements_path + 'processing_parameters.yaml'
    ftp_proc_parameters = yaml.safe_load(open(parameter_file))

    # Parameters for FTP filtering
    n          = ftp_proc_parameters['FTP_PROCESSING']['n']
    th         = ftp_proc_parameters['FTP_PROCESSING']['th']
    N_iter_max = ftp_proc_parameters['FRINGE_EXTRAPOLATION']['N_iter_max']
    L          = ftp_proc_parameters['FTP_PROJECTION']['L']
    D          = ftp_proc_parameters['FTP_PROJECTION']['D']
    pixel_size = ftp_proc_parameters['MEASUREMENT']['pixel_size']

    lin_min_idx = ftp_proc_parameters['MEASUREMENT']['lin_min_idx']
    lin_max_idx = ftp_proc_parameters['MEASUREMENT']['lin_max_idx']
    col_min_idx = ftp_proc_parameters['MEASUREMENT']['col_min_idx']
    col_max_idx = ftp_proc_parameters['MEASUREMENT']['col_max_idx']


    # FTP PROCESSING
    #logger.info('Loading gray and reference images')
    gray = np.load(measurements_path+'gray.npy')
    ref = np.load(measurements_path+'reference.npy')

    #Slin, Scol = np.shape(gray)
    Slin0, Scol0 = np.shape(gray)
    Slin = lin_max_idx - lin_min_idx
    Scol = col_max_idx - col_min_idx

    gray_resized = gray[lin_min_idx:lin_max_idx,col_min_idx:col_max_idx]
    gray_data_array = xr.DataArray(gray_resized, dims = ('x', 'y'))
    gray_data_array.to_netcdf(measurements_path + 'gray.nc')

    ref_resized = ref[lin_min_idx:lin_max_idx,col_min_idx:col_max_idx]
    ref_data_array = xr.DataArray(ref_resized, dims = ('x', 'y'))
    ref_data_array.to_netcdf(measurements_path + 'ref.nc')



    if masked==True:
        mask = np.load(measurements_path+'mask.npy')
        mask_resized = mask[lin_min_idx:lin_max_idx,col_min_idx:col_max_idx]
        mask_data_array = xr.DataArray(mask_resized, dims = ('x', 'y'))
        mask_data_array.to_netcdf(measurements_path + 'mask.nc')
    else:
        mask = 0


    def_files = natsorted(glob.glob(os.path.join(ftp_im_path, '*.bmp')), key=lambda y: y.lower())

    # MODIFICAR !!
    # def_files_cut = def_files[8873:]
    # def_files_cut.insert(0, def_files[0])
    # def_files = def_files_cut

    N_defs = len(def_files)


    logger.info('N_defs = ' + str(N_defs))
    #print('N_defs = ' + str(N_defs))

    if masked==True:
        resfactor = np.mean(ref*mask)/np.mean(gray*mask)
    else:
        resfactor = np.mean(ref)/np.mean(gray)

    # 7. Generate (referece-gray) image.
    ref_m_gray = ref - resfactor*gray

    # Calculate wavelength of the projected pattern
    line_ref = np.average(ref_m_gray, 0)
    peaks, _ = find_peaks(line_ref, height=0)

    wavelength_pix = np.mean(np.diff(peaks))

    pspp = pixel_size*wavelength_pix

    # logger.info('FTP processing')

    t1 = time.time()

    # Supress mean of the first image to all of them
    def_image = sio.imread(def_files[0])
    def_image = def_image.astype(float)
    def_m_gray = def_image - resfactor*gray
    
    if masked==True:
        def_m_gray = def_m_gray*mask + ref_m_gray*(1-mask)
        dphase0 = calculate_phase_diff_map_1D(def_m_gray, ref_m_gray, th, n, mask_for_unwrapping=(1-mask))
        dphase0 = np.mean(ma.masked_array(dphase0, mask=(1-mask))[lin_min_idx:lin_max_idx,col_min_idx:col_max_idx])
    else:  
        dphase0 = calculate_phase_diff_map_1D(def_m_gray, ref_m_gray, th, n)
        dphase0 = np.mean(dphase0[lin_min_idx:lin_max_idx,col_min_idx:col_max_idx]) 

    iteration_time = []
    for j in tqdm(range(int(N_defs/N_vertical_slices))):

        ti = time.time()
        p = Pool(cores)

        non_iterable_args = def_files, resfactor, gray, ref_m_gray, mask, N_iter_max, n, th, L, D, pspp, dphase0, lin_min_idx,lin_max_idx,col_min_idx,col_max_idx, ftp_nc_path
        p.starmap(analyze_with_ftp_and_save_to_nc, zip(np.arange(N_vertical_slices*j, N_vertical_slices*(j+1)), repeat(non_iterable_args)))

        p.close()
        gc.collect()

        tf = time.time() - ti
        iteration_time.append(tf)

    t2 = time.time()

    dt = t2 - t1
    #print(dt)
    print('Time for FTP analysis = '+str(round(dt,2)) + 's' + '\n')


    plt.figure()
    plt.plot(iteration_time,'.-')
    plt.xlabel('Number of iteration')
    plt.ylabel('Iteration time [s]')
    plt.title('FTP analysis')
    plt.grid()
    plt.show()


  # ## NON PARALLELIZED
    # accum = np.zeros((Slin, Scol, N_defs)) 
    # for k in tqdm(range(N_defs)):
    #     img = sio.imread(def_files[k])
    #     def_image = img.astype(float)
    #     def_m_gray = def_image - resfactor*gray
    #     def_m_gray = def_m_gray*mask + ref_m_gray*(1-mask)
    #     # 4. Process by FTP
    #     dphase = calculate_phase_diff_map_1D(def_m_gray, ref_m_gray, th, n, mask_for_unwrapping=(1-mask))
    #     accum[:,:,k] = dphase[lin_min_idx:lin_max_idx,col_min_idx:col_max_idx]
    # height_fields_dset[:, :, :] = accum


    return None
