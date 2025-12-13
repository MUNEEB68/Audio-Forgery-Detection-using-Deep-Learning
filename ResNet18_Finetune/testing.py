# testing.py 
import sys
import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import librosa

# Path to src folder for spectrogram_extraction
src_path = r"C:\Users\munee\Desktop\audio forgery detection\src"
sys.path.append(src_path)
from spectrogram_extraction import spectrogram_extraction

# ================= Dataset ================= #
class AudioDataset(Dataset):
    def __init__(self, audio_folder, label_dict, subset_ratio=1.0):
        self.audio_folder = audio_folder
        self.label_dict = label_dict
        self.files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]

        random.shuffle(self.files)

        # subset
        if subset_ratio < 1.0:
            subset_size = int(len(self.files) * subset_ratio)
            self.files = self.files[:subset_size]

        self.fake_count = sum(1 for f in self.files if label_dict[f] == 0)
        self.real_count = sum(1 for f in self.files if label_dict[f] == 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.audio_folder, file_name)

        audio, sr = librosa.load(file_path, sr=16000)

        spec = spectrogram_extraction(audio)

        # 3-channel
        spec = np.stack([spec, spec, spec], axis=0)
        spec = torch.tensor(spec, dtype=torch.float32)

        label = torch.tensor(self.label_dict[file_name], dtype=torch.long)
        return spec, label,file_name

# loading labels
def load_labels(label_file):
    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            parts = line.split()
            label_dict[parts[0]] = int(parts[1])
    return label_dict

######## MAIN ########
if __name__ == "__main__":
    logging.basicConfig(
        filename='test_log.txt',
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        filemode='w'
    )

    logging.info("Starting TEST script...")

    # Load labels
    label_file = r"test_target_labels.txt"
    label_dict = load_labels(label_file)
    logging.info(f"Loaded {len(label_dict)} labels (TEST SET)")

    # Paths and settings
    test_audio_folder = r"D:\data\HAD\HAD_test\test\test"
    subset_ratio = 1
    batch_size = 8
    save_folder = r"C:\Users\munee\Desktop\audio forgery detection\ResNet18_Finetune" #for checkpoint path

    # Dataset & DataLoader
    dataset = AudioDataset(test_audio_folder, label_dict, subset_ratio=subset_ratio)
    logging.info(f"TEST SET: {len(dataset)} samples")
    logging.info(f"Fake samples: {dataset.fake_count}, Real samples: {dataset.real_count}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    checkpoint_path = os.path.join(save_folder, "resnet18_epoch_9.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model = model.to(device)
    model.eval()

    # Create output file for FAKE classified samples
    output_file = "file_classified.txt"
    open(output_file, "w").close()  # clear file at start

    # Evaluation
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs, labels, file_names = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # ===== SAVE FAKE CLASSIFIED FILES =====
            for fname, pred_label in zip(file_names, predicted.cpu()):
                if pred_label.item() == 0:   # 0 = Fake
                    with open(output_file, "a") as f:
                        f.write(fname + "\n")

            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                batch_accuracy = 100 * correct / total
                logging.info(f"Batch {batch_idx + 1}: Running Accuracy = {batch_accuracy:.2f}%")

    accuracy = 100 * correct / total
    print(f'Test Set Accuracy: {accuracy:.2f}%')
    logging.info(f'Test Set Accuracy: {accuracy:.2f}%')

    print("Testing completed.")
    logging.info("Testing completed.")
