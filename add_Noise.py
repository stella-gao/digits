import numpy as np
import scipy.io.wavfile as wav
import scipy.io.wavfile as wav

from scipy import signal


# add_Noise: add noise to a given signal. 

def add_Noise(signal, k = 16.0) :
    
    N = len(signal)
    power = np.mean(signal**2)
    noise_power = k * power

    noise = np.array(np.random.normal(scale = np.sqrt(noise_power), size = N), dtype = np.int16)
    signal_noise = signal + noise

    return signal_noise



# Example:

fs, sig = wav.read('FAC_1A.wav')
sig_noise = add_Noise(sig, k = 25)

# Writing file.wav
wav.write('FAC_1A_noise.wav', fs, sig_noise)

