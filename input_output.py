import os
import glob
import numpy as np
import skimage.io as sio
import yaml
import time


import logging
logging.basicConfig()
logger = logging.getLogger(__name__)



def generate_average_gray_and_reference_images(measurements_path, cut=False):
    """
    Generates an average of all the gray and references images and saves it as .npy.

    Input:
    - measurements_path : string
    directory where all the measurements are (including gray and reference)
    - cut : bool, optional (default=False)
    Indicates if the array should be cut in size before saving, using the parameters specified in processing_parameters.yaml

    Output:
    Saves the aveages in the directory measurements_path.

    """

    logger.info('---------- AVERAGE GRAY AND REFERENCE ----------')
    parameter_file = measurements_path + 'processing_parameters.yaml'
    ftp_proc_parameters = yaml.safe_load(open(parameter_file))
    lin_min_idx = ftp_proc_parameters['MEASUREMENT']['lin_min_idx']
    lin_max_idx = ftp_proc_parameters['MEASUREMENT']['lin_max_idx']
    col_min_idx = ftp_proc_parameters['MEASUREMENT']['col_min_idx']
    col_max_idx = ftp_proc_parameters['MEASUREMENT']['col_max_idx']

    t1 = time.time()

    logger.info('Creating average gray image')
    gri_files = sorted(glob.glob(os.path.join(measurements_path, 'gray', '*.bmp')))

    image0 = sio.imread(gri_files[0])
    if 'cut'==True:
        image0 = image0[lin_min_idx:lin_max_idx,col_min_idx:col_max_idx]

    # Assign image size
    Slin, Scol = np.shape(image0)
    Ngri = len(gri_files)
    acum_gri = np.zeros((Slin, Scol))
    for kk in range(Ngri):
        image = sio.imread(gri_files[kk])
        image = image.astype(float)
        if 'cut'==True:
            image = image[lin_min_idx:lin_max_idx,col_min_idx:col_max_idx]

        acum_gri = acum_gri + image

    gri_average = acum_gri/Ngri
    np.save(measurements_path+'gray', gri_average)

    logger.info('Creating average reference image')
    ref_files = sorted(glob.glob(os.path.join(measurements_path, 'reference', '*.bmp')))
    Nref = len(ref_files)
    acum_ref = np.zeros((Slin, Scol))
    for kk in range(Nref):
        image = sio.imread(ref_files[kk])
        image = image.astype(float)
        if 'cut'==True:
            image = image[lin_min_idx:lin_max_idx,col_min_idx:col_max_idx]


        acum_ref = acum_ref + image

    ref_average = acum_ref/Nref
    np.save(measurements_path + 'reference', ref_average)

    t2 = time.time()

    dt = t2 - t1
    #print(dt)
    print('Time for average gray and reference images = '+str(round(dt,2)) + 's' + '\n')

    return None