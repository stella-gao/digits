from matplotlib import pyplot as plt
import numpy as np

from datasets import clean_digits
import features

(X_train, y_train), (X_test, y_test) = clean_digits.load_data()

x = X_train[0]
plt.figure()
plt.plot(x, 'k')
plt.axis('tight')

plt.xticks(np.arange(0, len(x), 8000 / 10), np.arange(0, len(x) / 8000.0, 0.1))
plt.xlabel('time (s)')
plt.ylabel('amplitude')

plt.savefig('report/waveform.png')

sg = features.spectrogram(x, 256, 32)
plt.figure()
plt.pcolormesh(sg, cmap='gray')
plt.axis('tight')

frames_per_second = 8000 / 32.0
plt.xlabel('time (s)')
plt.xticks(np.arange(0, sg.shape[1], frames_per_second / 10),
           np.arange(0, sg.shape[1] / frames_per_second, 0.1))
plt.ylabel('frequency (Hz)')
plt.yticks(np.arange(0, 128, 16), np.arange(0, 4000., 4000. * 16 / 128))

plt.savefig('report/spectrogram.png')

basis = features.mel_basis(128, 256, 8000)
plt.figure()
plt.pcolormesh(basis, cmap='gray')
plt.axis('tight')
plt.savefig('report/mel_basis.png')

melscale = np.dot(basis, sg)
plt.figure()
plt.pcolormesh(melscale, cmap='gray')
plt.axis('tight')

plt.xlabel('time (s)')
plt.xticks(np.arange(0, sg.shape[1], frames_per_second / 10),
           np.arange(0, sg.shape[1] / frames_per_second, 0.1))

plt.ylabel('mel frequency (log Hz)')
plt.yticks(np.arange(0, 128, 16),
           ["%0.2f" % x for x in features.hz2mel(np.arange(0, 4000., 4000. * 16 / 128))])

plt.savefig('report/mfsc.png')

#plt.show()
