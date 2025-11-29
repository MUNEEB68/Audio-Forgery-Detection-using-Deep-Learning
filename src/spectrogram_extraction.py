# spectrogram_extraction.py
import numpy as np
from scipy.signal import spectrogram, windows
import cv2

def spectrogram_extraction(audio, display=False):
    sampling_rate = 16000
    window_length = int(0.025 * sampling_rate)
    hop_length = window_length // 4
    alpha = 2.5
    frequency_bins = 512

    # Gaussian window
    window = windows.gaussian(window_length, std=alpha)

    # Pad if audio is too short
    if len(audio) < window_length:
        audio = np.pad(audio, (0, window_length - len(audio)), mode='constant')

    # Compute spectrogram
    f, t, S = spectrogram(audio, fs=sampling_rate, window=window, noverlap=window_length - hop_length, nfft=frequency_bins)
    
    # Log magnitude (dB)
    spec = 20 * np.log10(np.abs(S) + 1e-6)

    # Z-score normalization
    spec = (spec - np.mean(spec)) / np.std(spec)

    # Resize to 256x 256 for consistency
    spec_resized = cv2.resize(spec, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Optional display
    if display:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(spec, aspect='auto', origin='lower', cmap='jet')
        plt.colorbar()
        plt.title("Original Spectrogram")
        plt.show()

        plt.figure()
        plt.imshow(spec_resized, aspect='auto', origin='lower', cmap='jet')
        plt.colorbar()
        plt.title("Resized Spectrogram")
        plt.show()

    return spec_resized
