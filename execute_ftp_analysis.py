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

masked = True    
#masked = False 
periodicity= 24                                     

#measurements_path = '/Volumes/T7 Shield/2024-07-16/sweep-bateur-mur/'
#measurements_path = '/Volumes/T7 Shield/2024-08-29/floater5/'

#for f in range(4,5):
#folder_name = 'a7.5-f'+ str(f)

#folder_name = 'sweep-a6-f1-20-1min'
#folder_name = 'sweep-a400-f1-20-t60'



measurements_path = '/Volumes/T7 Shield/2024-09-30/'
folder_name = 'sweep-a1-f20-0-2min'


ftp_im_path = measurements_path + f'def-{folder_name}/'
print(folder_name)

#generate_average_gray_and_reference_images(measurements_path)

process_images_by_ftp(measurements_path, folder_name, ftp_im_path, masked=masked, N_vertical_slices=cores)

# Solid bottom, old code
#perform_fft_and_save_planes_solid(measurements_path, folder_name, periodicity=periodicity, N_vertical_slices=cores, plot_theorical_disp_relation=True)

# Azimuthal average, new code
save_lines_fft_averaged(measurements_path, folder_name)
