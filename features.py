import numpy as np
import scipy.signal as signal


def frame(x, frame_size, step):
    """
    Split vector into frames.

    Slices the input vector into "frames" - contiguous regions 
    beginning at multiples of step and of length frame_size.

    Returns a matrix of dimension (num_slices, frame_size) where
    each row is a frame.
    """
    num_frames = (len(x) - frame_size) // step
    framed = np.zeros((num_frames, frame_size))
    for frame_index in range(num_frames):        
        start = frame_index * step
        end = start + frame_size
        framed[frame_index, :] = x[start:end]
    return framed


def spectrogram(x, fft_size):
    framed = frame(x, fft_size, 1)
    window = signal.tukey(fft_size, 0.25)
    windowed = framed * window
    fft = np.fft.fft(windowed, axis=1)
    return np.log(np.abs(fft[:, 0:(fft.shape[1] // 2)].T))


def mel_basis(num_filters, fft_size, fs):
    """Create a basis for projecting an FFT to mel-frequency space."""

    def hz2mel(hz):
        return np.log(1.0 + hz / 700.0)

    def mel2hz(mel):
        return (np.exp(mel) - 1.0) * 700.0
    
    min_freq = 0
    max_freq = fs / 2.0

    min_mel = hz2mel(min_freq)
    max_mel = hz2mel(max_freq)
    mels = np.linspace(min_mel, max_mel, num=num_filters + 2)

    freqs = mel2hz(mels)

    # convert mel_freqs to FFT bins
    fft_freqs = freqs * (fft_size // 2) / max_freq

    # create matrix of projections.
    basis = np.zeros((num_filters, fft_size // 2))

    for i in range(num_filters):
        li = i
        ci = i + 1
        ri = i + 2
    
        # triangle height set so basis integrates to 1.0.
        h = 2.0 / (fft_freqs[ri] - fft_freqs[li])
    
        l_m = h / (fft_freqs[ci] - fft_freqs[li])
        l_b = -l_m * fft_freqs[li]
    
        r_m = -h / (fft_freqs[ri] - fft_freqs[ci])
        r_b = -r_m * fft_freqs[ri]
    
        for z in range(fft_size // 2):
            llb = np.maximum(fft_freqs[li], z)
            lrb = np.minimum(fft_freqs[ci], z + 1)        
            if llb < lrb:
                l_contribution = 0.5 * l_m * (lrb ** 2 - llb ** 2) + l_b * (lrb - llb)
                l_contribution = max(l_contribution, 0)
            else:
                l_contribution = 0
             
            rlb = np.maximum(fft_freqs[ci], z)
            rrb = np.minimum(fft_freqs[ri], z + 1)        
            if rlb < rrb:
                r_contribution = 0.5 * r_m * (rrb ** 2 - rlb ** 2) + r_b * (rrb - rlb)
                r_contribution = max(r_contribution, 0)
            else:
                r_contribution = 0
        
            basis[i, z] = l_contribution + r_contribution

    return basis


def mfsc(x, fft_size, num_filters, fs):
    sg = spectrogram(x, fft_size)
    basis = mel_basis(num_filters, fft_size, fs)
    return np.dot(basis, sg)
