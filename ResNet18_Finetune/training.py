# training.py
import sys
import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

        # Shuffle files to mix fake and real
        random.shuffle(self.files)

        # Take subset if subset_ratio < 1.0
        if subset_ratio < 1.0:
            subset_size = int(len(self.files) * subset_ratio)
            self.files = self.files[:subset_size]

        # Count fake and real
        self.num_fake = sum(1 for f in self.files if label_dict[f] == 0)
        self.num_real = sum(1 for f in self.files if label_dict[f] == 1)
        logging.info(f"Training subset contains {self.num_fake} fake files and {self.num_real} real files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.audio_folder, file_name)

        # Load audio
        audio, sr = librosa.load(file_path, sr=16000)

        # Convert to spectrogram
        spec = spectrogram_extraction(audio)  # shape: 256x256

        # Convert to 3-channel
        spec = np.stack([spec, spec, spec], axis=0)
        spec = torch.tensor(spec, dtype=torch.float32)

        # Get label
        label = torch.tensor(self.label_dict[file_name], dtype=torch.long)
        return spec, label

# ================= Labels ================= #
def load_labels(label_file):
    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_name = parts[0]
            label = int(parts[1])
            label_dict[file_name] = label
    return label_dict

# ================= Main ================= #
if __name__ == "__main__":
    # ================= Logging ================= #
    logging.basicConfig(
        filename='training_log.txt',
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        filemode='w'
    )

    logging.info("Starting training script...")

    # Load labels
    label_file = r"target_labels.txt"
    label_dict = load_labels(label_file)
    logging.info(f"Loaded {len(label_dict)} labels")

    # Print first/last 5 for sanity check
    logging.info("First 5 labels:")
    for k, v in list(label_dict.items())[:5]:
        logging.info(f"{k}: {v}")
    logging.info("Last 5 labels:")
    for k, v in list(label_dict.items())[-5:]:
        logging.info(f"{k}: {v}")

    # Paths
    train_audio_folder = r"D:\data\HAD\HAD_train\train\conbine"
    subset_ratio = 0.1  # use 10% of data for faster training
    batch_size = 8

    # Dataset & DataLoader
    train_dataset = AudioDataset(train_audio_folder, label_dict, subset_ratio=subset_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    logging.info(f"DataLoader created successfully. {len(train_dataset)} samples will be used.")

    # ================= Model ================= #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(pretrained=True)
    logging.info(model)

    # Make all layers trainable
    for param in model.parameters():
        param.requires_grad = True

    # Replace final fc layer
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    logging.info(f"Model initialized on {device}. All layers are trainable.")

    # ================= Loss & Optimizer ================= #
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # all layers being trained
    logging.info("Loss function and optimizer initialized")

    # ================= Training Loop ================= #
    num_epochs = 5
    total_samples_processed = 0
    logging.info(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        logging.info(f"Epoch {epoch+1} started")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_samples_processed += inputs.size(0)

            if (i + 1) % 10 == 0:
                avg_loss = running_loss / 10
                logging.info(f"[Epoch {epoch+1}, Batch {i+1}] avg loss: {avg_loss:.4f}, total samples processed: {total_samples_processed}")
                running_loss = 0.0

        logging.info(f"Epoch {epoch+1} completed, total samples processed so far: {total_samples_processed}")

    # ================= Save Model ================= #
    model_path = "resnet18_finetuned_audio.pth"
    torch.save(model.state_dict(), model_path)
    logging.info(f"Training complete. Model saved to {model_path}")
