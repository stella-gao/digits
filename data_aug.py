import numpy as np
    
def speed(data, factor):
	'''Augment a list of wav files and store in a 3D array'''
	shifted = np.zeros((len(data), 16000, 1))
	for i, wav in enumerate(data):
		indices = np.round(np.arange(0, len(wav), factor))
		indices = indices[indices < len(wav)].astype(int)
		shifted[i, 0:len(indices), 0] = wav[indices.astype(int)]
		
	return shifted
	

def stretch(data, factor, window_size, h):
	# Stretch the sound by a factor 
	result = np.zeros((len(data), 16000 / f + window_size, 1))
	for i, wav in enumerated(data):
		phase = np.zeros(window_size)
	    hanning_window = np.hanning(window_size)
	    out = np.zeros(len(wav) / f + window_size)
		for j in np.arange(0, len(wav)-(window_size+h), h * f):
			# two potentially overlapping subarrays
			a1 = wav[i : i + window_size]
			a2 = wav[i + h: i + window_size + h]
			# resynchronize the second array on the first 
			s1 = np.fft.fft(hanning_window * a1)
			s2 = np.fft.fft(hanning_window * a2)
			phase = (phase + np.angle(s2/s1)) % 2*np.pi
			a2_rephased = np.fft.ifft(np.abs(s2) * np.exp(1j * phase))
			# add to result
			i2 = int(i / f)
			out[i2 : i2 + window_size] += hanning_window * a2_rephased
		result[i, 0 : len(out), 0] = ((2 ** (16 - 4)) * out / out.max()) # normalise (16bit)
	return out
		
