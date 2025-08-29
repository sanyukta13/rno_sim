import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('/home/sany/rno_sim'))
from custom_funcs import *
from NuRadioReco.utilities import units, fft

# definitions
######################
c = 3*10**8  # speed of light in m/s
station_id = 11
wf_len = 2048
n_ice = 1.74
sampling_rate = 3.2 * units.GHz  # 3.2 GHz sampling rate

# files
######################
data = 'data/rno_cal5C_0dB.hdf5'
# ant_pkl = ''

#load files
######################
wf, time = read_hdf5(data)
wf = np.pad(wf, ((wf_len//2)-len(wf)//2, (wf_len//2)-len(wf)//2), mode='constant', constant_values=(wf[0]+wf[-1])/2)

peak_index = np.argmax(wf)
dt = 1/sampling_rate
time_array = np.arange(wf_len) * dt
time = time_array - time_array[peak_index]



