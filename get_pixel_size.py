import numpy as np
from matplotlib import pyplot as plt
import skimage.io as sio
from scipy.signal import find_peaks

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
#mpl.rcParams['text.usetex'] = True

plt.close('all')
plt.ion()

measurements_path = '/Volumes/T7/2025-04-04/'

image = sio.imread(measurements_path + 'setup.bmp')
image = image.astype(float)

# plt.figure()
# plt.imshow(image, aspect='auto', cmap='Grays')
# plt.colorbar()
# plt.show()

line = np.average(image, axis=0)
dline = np.abs(np.diff(line))
dline_cut = dline[500:1500]
peaks, _ = find_peaks(dline_cut, height=10, distance=3)

plt.figure()
plt.plot(dline, '.-')
plt.plot(peaks+500, dline_cut[peaks], '.')
plt.grid()
plt.show()

dist_2cm = np.mean(np.diff(peaks))

pixel_size = (20/dist_2cm)*1e-3 # m

print(pixel_size)
