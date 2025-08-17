
import matplotlib.pyplot as plt
plt.close('all')
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
mpl.rcParams['text.usetex'] = True
import cmocean
colormap = cmocean.cm.balance


from input_output import generate_average_gray_and_reference_images 
from bulk_processing import process_images_by_ftp
from post_processing import perform_fft_and_save_planes_solid
from azimuthal_average_fft import save_lines_fft_averaged
from ftp import height_map_from_phase_map
import os
import numpy as np
import numpy.ma as ma
import yaml
import skimage.io as sio
from tqdm import tqdm
from scipy.signal import find_peaks
from natsort import natsorted
import glob
import time
from scipy import signal
from scipy.signal.windows import tukey
from unwrap import unwrap

measurements_path = '/Volumes/T7/2025-04-04/'

folder_name = 'sweep-a100-f1-5-3min'
ftp_im_path = measurements_path + f'def-{folder_name}/'
print(folder_name)

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

gray = np.load(measurements_path+'gray.npy')
ref = np.load(measurements_path+'reference.npy')

#Slin, Scol = np.shape(gray)
Slin0, Scol0 = np.shape(gray)
Slin = lin_max_idx - lin_min_idx
Scol = col_max_idx - col_min_idx

def_files = natsorted(glob.glob(os.path.join(ftp_im_path, '*.bmp')), key=lambda y: y.lower())
N_defs = len(def_files)

resfactor = np.mean(ref)/np.mean(gray)

# 7. Generate (referece-gray) image.
ref_m_gray = ref - resfactor*gray

# Calculate wavelength of the projected pattern
line_ref = np.average(ref_m_gray, 0)
peaks, _ = find_peaks(line_ref, height=0)

wavelength_pix = np.mean(np.diff(peaks))

pspp = pixel_size*wavelength_pix


frame = 7200
th = 0.1




def_image = sio.imread(def_files[frame])
def_image = def_image.astype(float)
def_m_gray = def_image - resfactor*gray
ns = n

nx, ny = np.shape(gray)
lin = nx//2

# dY0 = ref_m_gray
# dY  = def_m_gray
# nx, ny = np.shape(dY)
# 
# lin = nx//2
# fY0=np.fft.fft(dY0[lin,:])
# fY=np.fft.fft(dY[lin,:])
# 
# dfy=1./ny
# fy=np.arange(dfy,1,dfy)
# 
# imax=np.argmax(np.abs(fY0[9:int(np.floor(ny/2))]))
# ifmax=imax+9
# 
# HW=np.round(ifmax*th)
# #W=2*HW
# W=2*HW+1
# win=signal.tukey(int(W),ns)
# 
# 
# #gaussfilt1D= np.zeros([1,nx])
# gaussfilt1D= np.zeros(ny)
# #gaussfilt1D[int(ifmax-HW-1):int(ifmax-HW+W-1)]=win
# gaussfilt1D[int(ifmax-HW):int(ifmax-HW+W)]=win
# 
# # Multiplication by the filter
# Nfy0 = fY0*gaussfilt1D
# Nfy = fY*gaussfilt1D
# 
# # Inverse Fourier transform of both images
# Ny0=np.fft.ifft(Nfy0)
# Ny=np.fft.ifft(Nfy)
# 
# phase0= np.unwrap(np.angle(Ny0))
# phase  = np.unwrap(np.angle(Ny))


#pixel_size = 0.4166666666666667/1000
xf = gray.shape[0]*pixel_size

fsize = 24

# RAW IMAGES

plt.figure()
plt.imshow(gray, aspect='equal', cmap = cmocean.cm.gray)#, extent=[0, xf, 0, xf])
plt.plot(np.linspace(0,nx, nx), np.ones(nx)*lin, c='firebrick')
plt.title('Gray', fontsize=fsize)
plt.xlim(0, nx)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(ref, aspect='equal', cmap = cmocean.cm.gray)#, extent=[0, xf, 0, xf])
plt.plot(np.linspace(0,nx, nx), np.ones(nx)*lin, c='firebrick')
plt.title('Reference', fontsize=fsize)
plt.colorbar()
plt.xlim(0, nx)
plt.show()


plt.figure()
plt.imshow(def_image, aspect='equal', cmap = cmocean.cm.gray)#, extent=[0, xf, 0, xf])
plt.plot(np.linspace(0,nx, nx), np.ones(nx)*lin, c='firebrick')
plt.title('Deformed', fontsize=fsize)
plt.colorbar()
plt.xlim(0, nx)
plt.show()



# LINES

plt.figure(figsize=(11,3))
plt.plot(gray[lin,:], c='gray', label='Gray')
plt.plot(ref[lin,:], c='royalblue', label='Reference')
plt.plot(def_image[lin,:], c='firebrick', label='Deformed')
plt.grid(alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1.0))
plt.tight_layout()
plt.show()



plt.figure(figsize=(11,3))
#plt.plot(gray[lin,:], c='gray', label='Gray')
plt.plot(ref_m_gray[lin,:], c='royalblue', label='Reference')
plt.plot(def_m_gray[lin,:], c='firebrick', label='Deformed')
plt.grid(alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1.0))
plt.tight_layout()
plt.show()

# FFT LINES

plt.figure(figsize=(4,4))
plt.plot(np.abs(np.fft.fft(ref[lin,:])), c='royalblue', label='Reference')
plt.plot(np.abs(np.fft.fft(def_image[lin,:])), c='firebrick', label='Deformed')
plt.grid(alpha=0.6)
plt.xlim(xmax=300)
plt.legend()
plt.tight_layout()
plt.title('Not supressing the gray')
plt.show()


plt.figure(figsize=(4,4))
#plt.plot(gray[lin,:], c='gray', label='Gray')
plt.plot(np.abs(np.fft.fft(ref_m_gray[lin,:])), c='royalblue', label='Reference')
plt.plot(np.abs(np.fft.fft(def_m_gray[lin,:])), c='firebrick', label='Deformed')
plt.grid(alpha=0.6)
plt.xlim(xmax=300)
plt.legend()
plt.title('Supressing the gray')
plt.tight_layout()
plt.show()

# FILTER

# Without supressing the gray
# fY0 = np.fft.fft(ref[lin,:])
# fY = np.fft.fft(def_image[lin,:])
# 
# imax=np.argmax(np.abs(fY0[9:int(np.floor(ny/2))]))
# ifmax=imax+9
# 
# HW=np.round(ifmax*th)
# W=2*HW+1
# win=tukey(int(W),ns)
# gaussfilt1D= np.zeros(ny)
# gaussfilt1D[int(ifmax-HW):int(ifmax-HW+W)]=win
# 
# g_without = gaussfilt1D
# 
# # Multiplication by the filter
# Nfy0 = fY0*gaussfilt1D
# Nfy = fY*gaussfilt1D

# Supressing the gray
fY0_mg = np.fft.fft(ref_m_gray[lin,:])
fY_mg  = np.fft.fft(def_m_gray[lin,:])


imax=np.argmax(np.abs(fY0_mg[9:int(np.floor(ny/2))]))
ifmax=imax+9

HW=np.round(ifmax*th)
W=2*HW+1
win=tukey(int(W),ns)
gaussfilt1D= np.zeros(ny)
gaussfilt1D[int(ifmax-HW):int(ifmax-HW+W)]=win

g_with = gaussfilt1D

# Multiplication by the filter
Nfy0_mg = fY0_mg*gaussfilt1D
Nfy_mg = fY_mg*gaussfilt1D

# plt.figure(figsize=(4,4))
# plt.plot(np.abs(fY0), c='royalblue', label='Reference')
# plt.plot(g_without*np.max(np.abs(fY0)), c='k', label='Filter')
# #plt.plot(np.abs(Nfy0), c='royalblue', label='Reference')
# #plt.plot(np.abs(Nfy), c='firebrick', label='Deformed')
# plt.grid(alpha=0.6)
# plt.xlim(xmax=300)
# plt.legend()
# plt.tight_layout()
# plt.title('Not supressing the gray')
# plt.show()


plt.figure(figsize=(4,4))
plt.plot(np.abs(fY0_mg), c='royalblue', label='Reference')
plt.plot(g_with*(np.abs(fY0_mg[ifmax])), c='k', label='Filter')
#plt.plot(np.abs(Nfy0_mg), c='royalblue', label='Reference')
#plt.plot(np.abs(Nfy_mg), c='firebrick', label='Deformed')
plt.grid(alpha=0.6)
plt.xlim(xmax=300)
plt.legend()
plt.title('Supressing the gray')
plt.tight_layout()
plt.show()

# plt.figure(figsize=(4,4))
# plt.plot(np.abs(Nfy0), c='royalblue', label='Reference')
# plt.plot(np.abs(Nfy), c='firebrick', label='Deformed')
# plt.grid(alpha=0.6)
# plt.xlim(xmax=300)
# plt.legend()
# plt.tight_layout()
# plt.title('Not supressing the gray')
# plt.show()


plt.figure(figsize=(4,4))
plt.plot(np.abs(Nfy0_mg), c='royalblue', label='Reference')
plt.plot(np.abs(Nfy_mg), c='firebrick', label='Deformed')
plt.grid(alpha=0.6)
plt.xlim(xmax=300)
plt.legend()
plt.title('Supressing the gray')
plt.tight_layout()
plt.show()


# DPHASE

# Without supressing the gray
# Inverse Fourier transform of both images
# Ny0 = np.fft.ifft(Nfy0)
# Ny  = np.fft.ifft(Nfy)
# 
# phase0 = np.unwrap(np.angle(Ny0))
# phase  = np.unwrap(np.angle(Ny))
# dphase = (phase-phase0)

#plt.figure()
#plt.plot(np.real(Ny0))
#plt.plot(np.real(Ny))
#plt.grid(alpha=0.6)
#plt.show()

# Supressing the gray
# Inverse Fourier transform of both images
Ny0_mg = np.fft.ifft(Nfy0_mg)
Ny_mg  = np.fft.ifft(Nfy_mg)

phase0_mg = np.unwrap(np.angle(Ny0_mg))
phase_mg  = np.unwrap(np.angle(Ny_mg))
dphase_mg = (phase_mg-phase0_mg)

plt.figure()
#plt.plot(dphase, c='royalblue', label='Not supressing the gray')
plt.plot(dphase_mg, c='gray', label='Supressing the gray')
plt.grid(alpha=0.6)
plt.legend()
plt.title('dphase')
plt.show()

height = height_map_from_phase_map(dphase_mg, L, D, pspp)

plt.figure(figsize=(7,4))
plt.plot(height*1e3, c='gray')
plt.grid(alpha=0.6)
plt.ylabel('$h$ [mm]')
plt.ylim([-1,1])
plt.title('n = '+str(n)+', th = ' +str(th))
plt.tight_layout()
plt.show()

ft = np.fft.fftshift(np.fft.fft(height))
kx = np.fft.fftshift(np.fft.fftfreq(len(height), pixel_size))

# Complete reldisp 
g = 9.82
gamma = 72e-3
rho = 1000
def f_reldisp(k, height):
    return np.sqrt((g*k + (gamma/rho)*k**3)*np.tanh(k*height))/(2*np.pi)

f = f_reldisp(kx, 3/100)

plt.figure()
plt.plot(f, np.abs(ft))
plt.grid()
plt.xlim([0,5])
plt.show()