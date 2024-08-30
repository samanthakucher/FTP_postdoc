import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from natsort import natsorted
import numpy.ma as ma
import glob
import os


import cmocean
colormap = cmocean.cm.balance

plt.close('all')
plt.ion()



measurements_path = '/Volumes/T7 Shield/2024-07-16/monochromatic/'  
folder_name = 'a7.5-f7'
ftp_nc_path = measurements_path + folder_name + '/' 


# gray  = xr.open_dataarray(measurements_path + 'gray.nc')
# ref   = xr.open_dataarray(measurements_path + 'ref.nc')
mask  = xr.open_dataarray(measurements_path + 'mask.nc')

height_fields_files = natsorted(glob.glob(os.path.join(ftp_nc_path, '*.nc')), key=lambda y: y.lower())

i = 0

h = xr.open_dataarray(height_fields_files[i]) 

plt.figure()
plt.imshow(ma.masked_array(h, mask=(1-mask)), aspect='equal', cmap=colormap)
#plt.imshow(h, aspect='equal', cmap=colormap)
plt.clim(-0.5,0.5)
plt.show()