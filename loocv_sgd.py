'''1D convolution example for spoken digits.

Modified from Keras MNIST CNN example.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

from datasets import clean_digits
import crop
import features
import wav_utils
import cross_validation

fft_size = 256
window_step = fft_size // 8
num_filters = 128
fs = 8000

# data params
nb_classes = 11

# training params
batch_size = 4
nb_epoch = 20

def cnn(train_speakers, test_speakers):
	# the data, shuffled and split between tran and test sets
	(X_train_wav, y_train), (X_test_wav, y_test) = cross_validation.load_data_set(train_speakers, test_speakers)

	shuf = np.random.permutation(len(X_train_wav))
	print(shuf)
	X_train_wav = [X_train_wav[i] for i in shuf]
	y_train = [y_train[i] for i in shuf]

	'''
	X_train_wav = X_train_wav[0:10]
	y_train = y_train[0:10]
	X_test_wav = X_test_wav[0:10]
	y_test = y_test[0:10]
	'''

	X_train_wav = crop.crop_list_arrays(X_train_wav, 1000, 0.05)
	X_test_wav = crop.crop_list_arrays(X_test_wav, 1000, 0.05)

	print(X_train_wav)

	max_length = max([len(x) for x in X_train_wav] + [len(x) for x in X_test_wav])

	print("normalizing...")
	X_train_wav = [wav_utils.normalize(x) for x in X_train_wav]
	X_test_wav = [wav_utils.normalize(x) for x in X_test_wav]                             

	print("padding to %d..." % max_length)
	X_train = wav_utils.pad_middle(X_train_wav, max_length)
	X_test = wav_utils.pad_middle(X_test_wav, max_length)

	assert not np.any(np.isnan(X_train))

	print("computing mfsc...")
	X_train = features.mfsc_matrix(X_train, fft_size, window_step, num_filters, fs)
	X_test = features.mfsc_matrix(X_test, fft_size, window_step, num_filters, fs)

	assert not np.any(np.isnan(X_train))

	print(X_train)

	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	model = Sequential()

	model.add(Convolution2D(nb_filter=32,
							nb_row=4,
							nb_col=4,
							input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
	model.add(Activation('relu'))

	model.add(Convolution2D(nb_filter=32,
							nb_row=4,
							nb_col=4))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 4)))
	model.add(Activation('relu'))

	model.add(Convolution2D(nb_filter=32,
							nb_row=4,
							nb_col=4))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Activation('relu'))

	model.add(Convolution2D(nb_filter=32,
							nb_row=4,
							nb_col=4))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Activation('relu'))

	model.add(Convolution2D(nb_filter=32,
							nb_row=4,
							nb_col=4))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Activation('relu'))


	'''
	model.add(Convolution2D(nb_filter=128,
							nb_row=3,
							nb_col=3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Activation('relu'))
	'''

	print(model.summary())

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd)

	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
			  show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
	score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	return (score[0], score[1])

speakers = [
    'MAE',
    'MBD',
    'MCB',
    'FAC',
    'FBH',
    'FCA',
    'MDL',
    'MEH',
    'FDC',
    'FEA']
    
scores = []
accuracies = []
    
for idx in range(len(speakers)):
	print(idx)
	train_speakers = speakers[:idx] + speakers[(idx+1):]
	test_speakers = []
	test_speakers.append(speakers[idx])
	score, accuracy = cnn(train_speakers, test_speakers)
	scores.append(score)
	accuracies.append(accuracy)
	
print(scores)
print(accuracies)
print(np.mean(accuracies))
