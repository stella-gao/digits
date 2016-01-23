'''1D convolution example for spoken digits.

Modified from Keras MNIST CNN example.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils import np_utils

import crop
from datasets import clean_digits
import wav_utils

# data params
nb_classes = 11

# training params
batch_size = 8
nb_epoch = 12

# the data, shuffled and split between tran and test sets
(X_train_wav, y_train), (X_test_wav, y_test) = clean_digits.load_data()

X_train_wav = crop.crop_list_arrays(X_train_wav, 1000, 0.05)
X_test_wav = crop.crop_list_arrays(X_test_wav, 1000, 0.05)

max_length = max([len(x) for x in X_train_wav] + [len(x) for x in X_test_wav])

X_train_wav = wav_utils.pad_middle(X_train_wav, max_length)
X_test_wav = wav_utils.pad_middle(X_test_wav, max_length)

def pack(X):
    result = np.zeros((len(X), len(X[0]), 1))
    for i in range(X.shape[0]):
        result[i, :, 0] = X[i]
    return result

X_train = pack(X_train_wav)
X_test = pack(X_test_wav)

assert not np.any(np.isnan(X_train))

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution1D(nb_filter=16,
                        filter_length=4,
                        input_shape=(max_length, 1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Activation('relu'))

model.add(Convolution1D(nb_filter=16,
                        filter_length=4))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Activation('relu'))

model.add(Convolution1D(nb_filter=16,
                        filter_length=4))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Activation('relu'))

model.add(Convolution1D(nb_filter=16,
                        filter_length=4))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Activation('relu'))

model.add(Convolution1D(nb_filter=16,
                        filter_length=4))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Activation('relu'))

model.add(Convolution1D(nb_filter=16,
                        filter_length=4))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Activation('relu'))

model.add(Convolution1D(nb_filter=16,
                        filter_length=4))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Activation('relu'))

model.add(Convolution1D(nb_filter=16,
                        filter_length=4))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Activation('relu'))

model.add(Convolution1D(nb_filter=16,
                        filter_length=4))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Activation('relu'))

print(model.summary())

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
