'''
Based on report found at
http://spoken-number-recognition.googlecode.com/svn/trunk/docs/Digit%20recognition/dsp_report.pdf

Uses k-NN (with k = 1) on the maximum of the MFCC components.
'''

import numpy as np
import scikits.talkbox.features as features

import datasets.clean_digits as clean_digits
import wav_utils as wav_utils

# the data, shuffled and split between tran and test sets
(X_train_wav, y_train), (X_test_wav, y_test) = clean_digits.load_data()

X_train_ceps = []
for wav in X_train_wav:
    ceps, _, _ = features.mfcc(wav, fs=8000, nceps=24)
    X_train_ceps.append(np.max(ceps, axis=0))
   

correct = 0.0
incorrect = 0.0
for wav, true_label in zip(X_test_wav, y_test):
    ceps, _, _ = features.mfcc(wav, fs=8000, nceps=24)
    mc = np.max(ceps, axis=0)

    min_diff = np.inf
    min_label = -1
    for train_mc, train_label in zip(X_train_ceps, y_train):
        diff = np.mean((mc - train_mc)**2)

        if diff < min_diff:
            min_diff = diff
            min_label = train_label

    if true_label == min_label:
        correct += 1
    else:
        incorrect += 1

print "correct: %d, incorrect: %d" % (correct, incorrect)
print "accuracy: %f" % (correct / (correct + incorrect))
        
