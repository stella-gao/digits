'''Based on keras examples/imdb_lstm.py.
'''

from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Permute, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb
from keras.optimizers import SGD

import crop
import datasets.clean_digits
import features
import wav_utils

fft_size = 256
window_step = fft_size // 8
num_filters = 128
fs = 8000

batch_size = 132

nb_classes = 11
nb_epoch = 100

print('Loading data...')
# the data, shuffled and split between tran and test sets
(X_train_wav, y_train), (X_test_wav, y_test) = datasets.clean_digits.load_data()

shuf = np.random.permutation(len(X_train_wav))
print(shuf)
X_train_wav = [X_train_wav[i] for i in shuf]
y_train = [y_train[i] for i in shuf]

'''
X_train_wav = X_train_wav[0:10]
y_train = y_train[0:10]
X_test_wav = X_test_wav[0:10]
y_test = y_test[0:10]
#'''

X_train_wav = crop.crop_list_arrays(X_train_wav, 1000, 0.05)
X_test_wav = crop.crop_list_arrays(X_test_wav, 1000, 0.05)

print(X_train_wav)

max_length = max([len(x) for x in X_train_wav] + [len(x) for x in X_test_wav])

print("normalizing...")
X_train_wav = [wav_utils.normalize(x) for x in X_train_wav]
X_test_wav = [wav_utils.normalize(x) for x in X_test_wav]                             

print("padding to %d..." % max_length)
X_train_wav = wav_utils.pad_left(X_train_wav, max_length)
X_test_wav = wav_utils.pad_left(X_test_wav, max_length)

print("computing mfsc...")
X_train_mfsc = features.mfsc_matrix(X_train_wav, fft_size, window_step, num_filters, fs)
X_test_mfsc = features.mfsc_matrix(X_test_wav, fft_size, window_step, num_filters, fs)

X_train = X_train_mfsc
X_test = X_test_mfsc

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('Build model...')

conv_features = 64

model = Sequential()

model.add(Convolution2D(nb_filter=conv_features,
                        nb_row=2,
                        nb_col=2,
                        input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))

model.add(Convolution2D(nb_filter=conv_features,
                        nb_row=2,
                        nb_col=2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 4)))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filter=conv_features,
                        nb_row=2,
                        nb_col=2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filter=conv_features,
                        nb_row=2,
                        nb_col=2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filter=conv_features,
                        nb_row=2,
                        nb_col=2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filter=conv_features,
                        nb_row=2,
                        nb_col=2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filter=conv_features,
                        nb_row=2,
                        nb_col=2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))

print(model.summary())

model.add(Permute((3, 1, 2)))
model.add(Reshape((45, conv_features * 1)))

#model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(256))  # try using a GRU instead, for fun
#model.add(Dropout(0.5))
model.add(Dense(11))
model.add(Activation('softmax'))

print(model.summary())

# try using different optimizers and different optimizer configs
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

print("Train...")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test), show_accuracy=True)
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)
