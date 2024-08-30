import numpy as np
from ftp import calculate_phase_diff_map_1D, height_map_from_phase_map
#from FTP_postdoc.ftp import calculate_phase_diff_map_1D, height_map_from_phase_map
#from FTP_postdoc.fringe_extrapolation import gerchberg2d
import numpy.ma as ma
from scipy.signal import medfilt
from unwrap import unwrap
import numpy.ma as ma
import xarray as xr
import dask.array as da
import skimage.io as sio


def analyze_with_ftp_and_save_to_nc(slice_index, args):

    def_files, resfactor, gray, ref_m_gray, mask, N_iter_max, n, th, L, D, pspp, dphase0, lin_min_idx,lin_max_idx,col_min_idx,col_max_idx, ftp_nc_path  = args

    img = sio.imread(def_files[slice_index])
    def_image = img.astype(float)

    def_m_gray = def_image - resfactor*gray
    if mask is not None and isinstance(mask, np.ndarray):
        def_m_gray = def_m_gray*mask + ref_m_gray*(1-mask)

    dphase = calculate_phase_diff_map_1D(def_m_gray, ref_m_gray, th, n) 
    dphase = unwrap(dphase)

    dphase = dphase - dphase0
    height = height_map_from_phase_map(dphase, L, D, pspp)
    h = xr.DataArray(height[lin_min_idx:lin_max_idx,col_min_idx:col_max_idx], dims=["x", "y"])
    h.to_netcdf(ftp_nc_path + str(slice_index) + '.nc')

    return None

def perform_fft2_per_frame_and_keep_solid_lines(kk, args):    

    height_fields_files, error_function, Scol, idx_kxB = args

    height = xr.open_dataarray(height_fields_files[kk]).values

    matrix = height*error_function
    new = 10
    m = np.fft.fftshift(np.fft.fft2(matrix, (matrix.shape[0]+2*new, matrix.shape[1]+2*new)))
    m = m[new: m.shape[0]-new, new: m.shape[1]-new]

    line_GX = m[:,Scol//2]
    line_XM = m[idx_kxB, :]
    line_MG = np.diagonal(m)

    return line_GX, line_XM, line_MG


def slice_fft2(kk, args):
    # vol, error_function = args
    # masked_frame = vol[:,:,kk]

    height_fields_files, ftp_nc_path, error_function = args

    height = xr.open_dataarray(height_fields_files[kk]) 

    matrix = height*error_function
    new = 10
    m = np.fft.fftshift(np.fft.fft2(matrix, (matrix.shape[0]+2*new, matrix.shape[1]+2*new)))
    m = m[new: m.shape[0]-new, new: m.shape[1]-new]
    return m
