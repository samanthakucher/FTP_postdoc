import numpy as np
from tqdm import tqdm 
import multiprocessing
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, interp2d
import h5py
import scipy.special as scp
import scipy.fftpack as scf 
import xarray as xr
from natsort import natsorted
import glob
import os
import yaml

import matplotlib.pyplot as plt

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

""" MASK + PADDING """
xa,xb = 30,30
xc,xd = 30,30

N_crop = 10

def set_fourier_domain(folder):
    """return the needed fourier abscisse in k, kx, ky ... """

    height_fields_files = natsorted(glob.glob(os.path.join(folder, '*.nc')), key=lambda y: y.lower())
    
    h_init = xr.open_dataarray(height_fields_files[0]).values
    
    h_map = h_init#[xa:-xb,xc:-xd] # Remove 10 points around the edge of the image
    
    nx,ny = h_map.shape
    #Nx,Ny = h_map.shape
    #nx,ny = 2*(Nx//N_crop), 2*(Ny//N_crop)

    #if nx pair
    h_map = h_map[:-1,:-1]
    nx,ny = h_map.shape
    
    midx, midy = nx//2, ny//2
    
    kx = np.arange(0,nx,1) - midx
    ky = np.arange(0,ny,1) - midy
    
    t = np.linspace(0, 2 * np.pi, 129)
    
    """ do the polar average along the correct radius, meaning in a circle of a size smaller than the rectangle image """
    if nx<ny:
        mid_i = midx
        k_i = kx
        k_ = k_i[mid_i:]
        k, theta = np.meshgrid(k_i[mid_i:], t[:128], indexing = 'ij')

    elif nx>ny:
        mid_i = midy
        k_i = ky
        k_ = k_i[mid_i:]
        k, theta = np.meshgrid(k_i[mid_i:], t[:128], indexing = 'ij')
        
    else:
        mid_i = midy
        k_i = ky
        k_ = k_i[mid_i:]
        k, theta = np.meshgrid(k_i[mid_i:], t[:128], indexing = 'ij')
            
    kxp = k * np.cos(theta)
    kyp = k * np.sin(theta)   

    dtheta = t[1] - t[0]
    
    return kx,ky,kxp,kyp,k_,dtheta




def polar_average(FFT_2D,kx,ky,kxp,kyp,k_,dtheta,pix_):
    
    if np.iscomplexobj(FFT_2D):    
        data_real = np.real(FFT_2D)
        data_imag = np.imag(FFT_2D)
        
        interpolator_real = RegularGridInterpolator((kx, ky), data_real, bounds_error=False, fill_value=0)
        interpolator_imag = RegularGridInterpolator((kx, ky), data_imag, bounds_error=False, fill_value=0)
    
        # Perform the interpolation
        points = np.vstack([kxp.ravel(), kyp.ravel()]).T
        spp_real = interpolator_real(points)
        spp_imag = interpolator_imag(points)
        
        spp_complex = spp_real + 1j * spp_imag
        spp2 = np.reshape(spp_complex, (k_.shape[0], 128))
        
        Nk = len(k_)
        dk = 2 * np.pi / (pix_ * 2 * Nk)
        k__ = k_ + (k_[1]-k_[0])/2
        spp_averaged = np.sum(spp2, axis=1) * dtheta * k__ * dk # dk donne la bonne dimension a integration en theta 
    else:
        raise TypeError("The input array being 2D FFT must be an array of complex data")
    
    return spp_averaged


def spectre_2D(args):
    folder, j, kx, ky, kxp, kyp, k_, dtheta, pix_ = args
    #ftp_nc_path, j, kx, ky, kxp, kyp, k_, dtheta, pix_ = args
    height_fields_files = natsorted(glob.glob(os.path.join(folder, '*.nc')), key=lambda y: y.lower())
    
    h_map = xr.open_dataarray(height_fields_files[j]).values

    nx,ny = h_map.shape

    #if nx pair
    h_map = h_map[:-1,:-1]
    nx,ny = h_map.shape
    
    
    h_map[:xa,:] = 0
    h_map[nx-xa:,:] = 0
    h_map[:,:xa] = 0
    h_map[:,ny-xa:] = 0
    
    h_map = h_map - np.mean(h_map)
    
    hist, bin_edge = np.histogram(h_map, bins=50)
    bin_center = (bin_edge[:-1]+bin_edge[1:])/2
    std_map = np.std(h_map)
    
    """ erf filter """
    xc = nx//2
    yc = ny//2
    radius = min(nx,ny)//2
    xx,yy = np.arange(nx),np.arange(ny)
    xi,yi = np.meshgrid(xx,yy,indexing='ij')
    error_function = (1 - scp.erf( (xi-xc)**2 + (yi-yc)**2 - radius**2 ))/2
    h_map_window_erf = h_map * error_function

    """ Hamming in space - ~bof """
    window_x = np.blackman(nx)
    window_y = np.blackman(ny)
    window_xy = np.outer(window_x, window_y)
    h_map_window_xy = h_map_window_erf * window_xy
    
    """ 2D fft """

    # fft2 = np.fft.fft2( h_map_window_xy ) # documentation zero padding 
    # fft2_shift = np.fft.fftshift(fft2)
    
    # fft2 = scf.fft2( h_map_window_xy ) * pix_**2 # pour avoir la bonne unite dans la TF 2D
    # without window
    fft2 = scf.fft2( h_map ) * pix_**2 # pour avoir la bonne unite dans la TF 2D
    fft2_shift = scf.fftshift( fft2 )
    
    #nx_c,ny_c = nx//N_crop, ny//N_crop # crop of the 2D fft 
    #fft2_crop = fft2[nx//2-nx_c:nx//2+nx_c,ny//2-ny_c:ny//2+ny_c] # crop in Fourier space
    
    fft2_x = fft2_shift[:,yc]
    fft2_y = fft2_shift[xc,:]

    fft_k = polar_average(fft2_shift, kx, ky, kxp, kyp, k_, dtheta, pix_)
    
    return j, hist, bin_center, std_map, fft2_x, fft2_y, fft_k, fft2_shift



if __name__ == '__main__':
    measurements_path = '/Volumes/T7 Shield/2024-07-16/sweep-bateur-mur/'
    folder_name = 'sweep-a400-f1-20-t60'
    ftp_im_path = measurements_path + f'def-{folder_name}/'
    print(folder_name)

    height_fields_files = natsorted(glob.glob(os.path.join(folder, '*.nc')), key=lambda y: y.lower())
    N_images = len(height_fields_files)
    
    h_map = xr.open_dataarray(height_fields_files[0]).values

    #if nx pair
    h_map = h_map[:-1,:-1]
    nx,ny = h_map.shape

    hist, bin_edge = np.histogram(h_map, bins=50) # compute pdf size
    bin_center = (bin_edge[:-1]+bin_edge[1:])/2
    
    Nx,Ny = h_map.shape
    
    # declare arrays
    #histogram_k = np.zeros((int(min(Nx,Ny)/2), N_images))
    bin_k = np.zeros((len(bin_center), N_images))
    pdf_k = np.zeros((len(hist), N_images))
    std_k = np.zeros((N_images,))
    histogram_x = np.zeros((Nx, N_images), dtype=np.complex64)
    histogram_y = np.zeros((Ny, N_images), dtype=np.complex64) 
    spectre_xy = np.zeros((Nx, Ny), dtype=np.complex64) 
    
    num_processes = 10 # multiprocessing.cpu_count() - 1  # Number of CPU cores - 1
    pool = multiprocessing.Pool(processes=num_processes)
    
    kx,ky,kxp,kyp,k_,dtheta = set_fourier_domain(folder)
    items = [(folder, item_k, kx, ky, kxp, kyp, k_, dtheta,pix_) for item_k in range(0, N_images)]
    
    Nk = len(k_)
    dk = 2 * np.pi / (pix_ * 2 * Nk)
    k__ = k_ * dk
    np.save(folder_save+'array_k.npy', k__)

    with tqdm(total=N_images, desc="Recording Arrays") as pbar:
        # Process arrays in parallel

        for result in pool.imap(spectre_2D, items):
            j, hist, bin_center, std_map, psd_x, psd_y, FFT_k, spectre_xy = result
            bin_k[:,j-1] = bin_center
            pdf_k[:,j-1] = hist
            std_k[j-1] = std_map
            histogram_x[:, j-1] = psd_x
            histogram_y[:, j-1] = psd_y
            spectre_xy+=spectre_xy
            
            np.save(folder_save+f'fft_k_{j}.npy',FFT_k)
            pbar.update(1)

    pool.close()
    pool.join()
    
    #return bin_k, pdf_k, std_k, histogram_x, histogram_y, spectre_xy/N_images




    #def save_lines_fft_averaged(measurements_path, folder_name):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info('---------- FOURIER TRANSFORM ----------')

    folder_h = measurements_path+f'{folder_name}/'
    folder_fft = measurements_path+f'fft-k-{folder_name}/'
    if not os.path.exists(folder_fft):
        logger.info('Create new directory for averaged FFT lines')
        os.makedirs(folder_fft)

    parameter_file = measurements_path + 'processing_parameters.yaml'
    ftp_proc_parameters = yaml.safe_load(open(parameter_file))
    pixel_size = ftp_proc_parameters['MEASUREMENT']['pixel_size']
    pixel_size = pixel_size/1000

    dict_ = dict()

    dict_["bin_t"], dict_["pdf_t"], dict_["std_t"], dict_["psd_x"], dict_["psd_y"], dict_["spectre_kxky"] = main(folder_h,folder_fft, pixel_size)
    
    np.save(measurements_path+f'data_vs_t_{folder_name}',dict_) 

    #return None

# if __name__ == '__main__':
#     
#     pix = 0.416/1000
#     folder = '/Volumes/T7 Shield/2024-08-24/sweep/'
#     name_gen = 'sweep-a6.5-f13-20-1min'
#     folder_h = folder+f'{name_gen}/'
#     folder_fft = folder+f'fft-k-{name_gen}/'
#     dict_ = dict()
# 
#     dict_["bin_t"], dict_["pdf_t"], dict_["std_t"], dict_["psd_x"], dict_["psd_y"], dict_["spectre_kxky"] = main(folder_h,folder_fft, pix)
#     
#     np.save(folder+f'data_vs_t_{name_gen}',dict_)
#     
#     # check code:
#     # j = 3000
#     # kx, ky, kxp, kyp, k_, dtheta = set_fourier_domain(folder_h)
#     # arg = folder_h, j, kx, ky, kxp, kyp, k_, dtheta
#     # j, hist, bin_center, std_map, fft2_x, fft2_y, fft_k = spectre_2D(arg)
#     
#     
#     # plt.figure()
#     # plt.imshow(np.log(abs(fft_k)),cmap='turbo')
#     # # plt.loglog(k_, abs(fft_k)**2) 
#     # plt.colorbar()
#     # plt.show()
    
