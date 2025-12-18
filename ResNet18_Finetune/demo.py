import sys
import os
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

        # 3-channel spectrogram
        spec = np.stack([spec, spec, spec], axis=0)
        spec = torch.tensor(spec, dtype=torch.float32)

        label = torch.tensor(self.label_dict[file_name], dtype=torch.long)
        return spec, label, file_name


# ================= Load labels ================= #
def load_labels(label_file):
    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            fname, lbl = line.split()
            label_dict[fname] = int(lbl)
    return label_dict


# ================= MAIN ================= #
if __name__ == "__main__":

    print("\n===== DEMO TESTING SCRIPT =====\n")

    # Load labels
    label_file = r"test_target_labels.txt"
    label_dict = load_labels(label_file)
    print(f"Loaded {len(label_dict)} labels")

    test_audio_folder = r"D:\data\HAD\HAD_test\test\test"
    batch_size = 8
    save_folder = r"C:\Users\munee\Desktop\audio forgery detection\ResNet18_Finetune"

    dataset = AudioDataset(test_audio_folder, label_dict)
    print(f"Total samples in dataset: {len(dataset)}")
    print(f"Fake samples: {dataset.fake_count}, Real samples: {dataset.real_count}\n")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint_path = os.path.join(save_folder, "resnet18_epoch_9.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.to(device)
    model.eval()

    print("\nModel loaded successfully.")
    print("\n===== RUNNING DEMO (5 BATCHES ONLY) =====\n")

    correct = 0
    total = 0
    max_demo_batches = 10

    with torch.no_grad():
        for batch_idx, (inputs, labels, file_names) in enumerate(loader):

            if batch_idx == max_demo_batches:
                break  # STOP AFTER 5 BATCHES

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)

            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)

            correct += batch_correct
            total += batch_total

            batch_acc = 100 * batch_correct / batch_total

            print(f"--- Batch {batch_idx + 1} ---")
            for fname, t, p in zip(file_names, labels.cpu(), predicted.cpu()):
                label_name_t = "Fake" if t.item() == 0 else "Real"
                label_name_p = "Fake" if p.item() == 0 else "Real"
                print(f"File: {fname} | Target: {label_name_t} | Predicted: {label_name_p}")

            print(f"Batch Accuracy: {batch_acc:.2f}%\n")

    final_acc = 100 * correct / total
    print("===== DEMO RESULTS =====")
    print(f"Total samples processed: {total}")
    print(f"Final Demo Accuracy: {final_acc:.2f}%")
    print("Demo testing completed.\n")
