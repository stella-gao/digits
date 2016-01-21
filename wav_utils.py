import numpy as np

def pad(data, padded_length):
    '''Zero-pad a list of wav files and store in 3D array.'''
    padded = np.zeros((len(data), padded_length, 1))
    for i, wav in enumerate(data):
        padded[i, :, 0] = np.concatenate(
            (wav, np.zeros(padded_length - wav.shape[0])))
    return padded
