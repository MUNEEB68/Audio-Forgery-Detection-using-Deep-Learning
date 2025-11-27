import torch
from torch.utils.data import Dataset, DataLoader
import os
from scipy.io import wavfile

import os
from torch.utils.data import Dataset
from scipy.io import wavfile

import os
from torch.utils.data import Dataset
from scipy.io import wavfile

class AudioDataset(Dataset):
    def __init__(self, folder_path, label_file, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        
        # List all wav files
        self.audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        # Load labels from the separate file
        with open(label_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f]  # convert to int

        assert len(self.audio_files) == len(self.labels), "Number of audio files and labels must match"

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        fs, audio = wavfile.read(os.path.join(self.folder_path, audio_file))
        audio = audio.astype(float)

        if self.transform:
            audio = self.transform(audio)

        return audio, label

