import gzip
from scipy.io import wavfile
import sys

from keras.datasets.data_utils import get_file

def load_speaker_set(speaker_set):

    filename_pattern = "%(speaker)s_%(digit)s%(repetition)s.wav"
    url_pattern = "http://www.ee.columbia.edu/~dpwe/sounds/tidigits/%(speaker)s/" + filename_pattern
    
    digits = [
        'O',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        'Z']

    repetitions = ['A', 'B']

    X = []
    y = []
    
    for speaker in speaker_set:
        for digit_ind, digit in enumerate(digits): 
            for repetition in repetitions:
                       
                filename = filename_pattern % locals()
                url = url_pattern % locals()
            
                path = get_file(filename, origin=url)
                    
                _fq, samples = wavfile.read(path)

                # Check that there is only one channel.
                assert samples.ndim == 1
                
                X.append(samples)
                y.append(digit_ind)
                
    return X, y


def load_data():

    train_speakers = [
        'MAE',
        'MBD',
        'MCB',
        'FAC',
        'FBH',
        'FCA']
    
    test_speakers = [
        'MDL',
        'MEH',
        'FDC',
        'FEA']

    X_train, y_train = load_speaker_set(train_speakers)
    X_test, y_test = load_speaker_set(test_speakers)
                        
    return ((X_train, y_train), (X_test, y_test))
