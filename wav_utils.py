import numpy as np


def pad_right(data, padded_length):
    '''Zero-pad a list of vectors by adding zeros to the end.'''
    padded = np.zeros((len(data), padded_length))
    for i, wav in enumerate(data):
        padded[i, :] = np.concatenate(
            (wav, np.zeros(padded_length - wav.shape[0])))
    return padded


def pad_left(data, padded_length):
    '''Zero-pad a list of vectors by adding zeros to the start.'''
    padded = np.zeros((len(data), padded_length))
    for i, wav in enumerate(data):
        padded[i, :] = np.concatenate(
            (np.zeros(padded_length - wav.shape[0]),
             wav))
    return padded


def pad_middle(data, padded_length):
    '''Zero-pad a list of vectors by adding an equal number of zeros to each end.'''
    padded = np.zeros((len(data), padded_length))
    for i, wav in enumerate(data):
        padded[i, :] = np.concatenate((
            np.zeros((padded_length - wav.shape[0] + 1) // 2),
            wav,
            np.zeros((padded_length - wav.shape[0]) // 2)))
    return padded


def normalize(x):
    x = np.array(x, dtype=np.float64)
    x_z = x - np.mean(x)
    return x_z / np.sqrt(np.mean(x_z ** 2))
