import numpy as np
import multiprocessing
multiprocessing.set_start_method("fork")
from multiprocessing import cpu_count

from input_output import generate_average_gray_and_reference_images 
from bulk_processing import process_images_by_ftp
from post_processing import perform_fft_and_save_planes_solid
from azimuthal_average_fft import save_lines_fft_averaged

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

cores = cpu_count()
cores = 10

import os
# Set NUMEXPR_MAX_THREADS to 10
os.environ["NUMEXPR_MAX_THREADS"] = "10"

#masked = True    
masked = False 
periodicity= 40/1000                                    


measurements_path = '/Volumes/T7/2025-04-17/10mm/' 


#folder_name = 'sweep-a100-f1-5-3min'
names =['sweep-a60-f1-5-3min', 'sweep-a80-f1-5-3min', 'sweep-a100-f1-5-3min', 'sweep-a120-f1-5-3min', 'sweep-a140-f1-5-3min', 'sweep-a160-f1-5-3min', 'sweep-a180-f1-5-3min','sweep-a200-f1-5-3min', 'sweep-a220-f1-5-3min', 'sweep-a240-f1-5-3min'] 

names = ['sweep-a60-f1-5-3min', 'sweep-a100-f1-5-3min']

ii = 0
for folder_name in names:
    #measurements_path = '/Volumes/T7/2025-01-30/h1_' + str(h1) + '/' 
    #for ii in range(len(hs)):
    #big_folder_name = str(hs[ii]) + 'mm'
    #for f in fs:
    #folder_name = 'a1-f'+str(f)
    #ftp_im_path = measurements_path + f'def-{folder_name}/'
    #measurements_path = path + big_folder_name + '/'
    ftp_im_path = measurements_path + f'def-{folder_name}/'

    print(folder_name)
    if ii==0:
        generate_average_gray_and_reference_images(measurements_path)

    process_images_by_ftp(measurements_path, folder_name, ftp_im_path, masked=masked, N_vertical_slices=cores)

    # Periodic bottom
    # perform_fft_and_save_planes_solid(measurements_path, folder_name, periodicity=periodicity, N_vertical_slices=cores, plot_theorical_disp_relation=False)

    # Azimuthal average
    # save_lines_fft_averaged(measurements_path, folder_name)
    ii = ii+1
