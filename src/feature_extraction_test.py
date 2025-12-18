from spectrogram_extraction import spectrogram_extraction
from scipy.io import wavfile

audio_file = r"D:\data\HAD\HAD_test\test\test\HAD_test_00000001.wav"
fs, audio = wavfile.read(audio_file)
spec = spectrogram_extraction(audio, display=True)
print(spec.shape)
