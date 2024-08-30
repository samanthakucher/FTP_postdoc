import multiprocessing
multiprocessing.set_start_method("fork")
from multiprocessing import cpu_count

from input_output import generate_average_gray_and_reference_images 
from bulk_processing import process_images_by_ftp
from post_processing import perform_fft_and_save_planes_solid

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

cores = cpu_count()
cores = 10

import os
# Set NUMEXPR_MAX_THREADS to 10
os.environ["NUMEXPR_MAX_THREADS"] = "10"

masked = True    
periodicity= 24                                     
                            
measurements_path = '/Volumes/T7 Shield/2024-08-29/floater35/'  
#for f in range(5,18):
fs = [4, 10,12,14,16]
for f in fs:
    folder_name = 'a6-f'+ str(f)
    #folder_name = 'sweep-a6-f1-20-1min'
    ftp_im_path = measurements_path + 'def-' + folder_name + '/'
    print(folder_name)


    #generate_average_gray_and_reference_images(measurements_path)

    process_images_by_ftp(measurements_path, folder_name, ftp_im_path, masked=masked, N_vertical_slices=cores)

    #perform_fft_and_save_planes_solid(measurements_path, folder_name, periodicity=periodicity, N_vertical_slices=cores, plot_theorical_disp_relation=True)