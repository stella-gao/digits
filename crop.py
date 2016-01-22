import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
from numpy.lib import stride_tricks


# Import 
fs, sig = wav.read('FAC_1A.wav')
fs2, sig2 = wav.read('FDC_1A.wav')
fs3, sig3 = wav.read('MBD_4A.wav')

# plot first audio file
plt.plot(sig)
plt.show()

# plot first spectrogram
f, t, Sxx = signal.spectrogram(sig, fs)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec] (?)')
plt.show()

# plot sprectogram in greyscale, and crop
Sxx=Sxx[0:80]

# plt.pcolormesh(t, f[0:80], Sxx, cmap = cm.Greys_r)
plt.pcolormesh(t, f[0:80], Sxx, cmap = 'gray')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec] (?)')
plt.show()

# function to return timepoints to crop a single list/np.array
def crop_timepoints_1_array(np1Darray, win_size, thresh_ratio):
    # np1Darray is a list of data, win_size is the moving average window size
    # thresh_ratio is the % of max to take as cut-off
    
    # absolute values (so moving average stays positive)
    abs_val = [abs(x) for x in np1Darray]
     
    # take a moving average with window size win_size
    moving_avg = np.convolve(abs_val,  np.ones((win_size,))/win_size, mode='valid')
    
    # find threshold
    threshold = max(moving_avg) * thresh_ratio
    
    # find timepoint larger than threshold and last time point smaller than threshold 
    first = next(x[0] for x in enumerate(moving_avg) if x[1] > threshold)
    last = next(x[0] for x in enumerate(reversed(moving_avg)) if x[1] > threshold)

    return first, len(moving_avg) - last + win_size

# test crop_timepoints_1_array and plot results
plt.plot(sig)
first, last = crop_timepoints_1_array(sig, 1000, 0.05)
plt.axvline(first)
plt.axvline(last)
plt.show()

# function to take .wav, find cropping time points and print
def plot_crop_points_file(wfile, win_size, thresh_ratio):
    
    fs, sig = wav.read(wfile) 

    plt.plot(sig)
    first, last = crop_timepoints_1_array(sig, win_size, thresh_ratio)
    plt.axvline(first)
    plt.axvline(last)
    plt.show()
    return

plot_crop_points_file('MBD_4A.wav', 1000, 0.05)

# function to take a waveform (already read in), find cropping time points and print
def plot_crop_points_sig(sig, win_size, thresh_ratio):
    
    plt.plot(sig)
    first, last = crop_timepoints_1_array(sig, win_size, thresh_ratio)
    plt.axvline(first)
    plt.axvline(last)
    plt.show()
    return

plot_crop_points_sig(sig, 1000, 0.05)

# function to apply crop_timepoints_1_array to each item of a list
def crop_timepoints_list_arrays(list_array, win_size, thresh_ratio):
    cropped = []
    for array in list_array:
        cropped.append(crop_timepoints_1_array(array, win_size, thresh_ratio))
    
    return cropped

# function to crop an array at the timepoints
def crop_1_array(array, win_size, thresh_ratio):
    
    first, last = crop_timepoints_1_array(array, win_size, thresh_ratio)
    out = array[first:last]
    return out

sig_crop = crop_1_array(sig, 1000, 0.05)

print(crop_timepoints_1_array(sig, 1000, 0.05))
plt.plot(sig_crop)
plt.show()

# apply crop_1_array to each item of a list
def crop_list_arrays(list_array, win_size, thresh_ratio):
    cropped = []
    for array in list_array:
        cropped.append(crop_1_array(array, win_size, thresh_ratio))
    
    return cropped

# test
sig_list = [sig, sig2, sig3]

crop_timepoints_list_arrays(sig_list, 1000, 0.05)

cropped_list = crop_list_arrays(sig_list, 1000, 0.05)

plt.plot(cropped_list[2])
plt.show()

sig_crop = crop_1_array(sig, 1000, 0.05)

print(crop_timepoints_1_array(sig, 1000, 0.05))
plt.plot(sig_crop)
plt.show()
plt.plot(sig[933:4633])
plt.show()

