
# Binary Fall Detection (Falling vs Non-Falling)
# Dataset = 10fps pose keypoints
# Train set is augmented, Test set untouched
# Softmax probabilities for 2 classes

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import random
import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------- DATASET -------------------
class PoseDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        pose_cols = [c for c in df.columns if c.endswith(('_x', '_y', '_c'))]
        assert len(pose_cols) == 51

        self.features = df[pose_cols].values.astype(np.float32)

        # Binary mapping
        class_map = {
            'Non-Falling': 0,
            'Falling': 1
        }

        labels_series = df['final_class'].map(class_map)

        if labels_series.isnull().any():
            raise ValueError("Unknown class found in CSV")

        self.labels = labels_series.values.astype(np.int64)

        self.sequences = []
        self.seq_labels = []
        self.lengths = []

        for video_path in df['video_path'].unique():
            mask = df['video_path'] == video_path
            video_features = self.features[mask]
            video_labels = self.labels[mask]

            if len(video_features) == 0:
                continue

            current_label = video_labels[0]
            start_idx = 0

            for i in range(1, len(video_labels)):
                if video_labels[i] != current_label:

                    if i - start_idx >= 5:
                        seq = video_features[start_idx:i]
                        self.sequences.append(torch.tensor(seq))
                        self.seq_labels.append(int(current_label))
                        self.lengths.append(i - start_idx)

                    current_label = video_labels[i]
                    start_idx = i

            if len(video_labels) - start_idx >= 5:
                seq = video_features[start_idx:]
                self.sequences.append(torch.tensor(seq))
                self.seq_labels.append(int(current_label))
                self.lengths.append(len(video_labels) - start_idx)

        print(f"{csv_file} → {len(self.sequences)} sequences")
        print("Class distribution:", np.bincount(self.seq_labels))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.seq_labels[idx], self.lengths[idx]


def collate_fn(batch):
    seqs, labels, lengths = zip(*batch)
    seqs = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    return seqs, torch.tensor(labels), torch.tensor(lengths)


# ------------------- MODEL -------------------
class FallingBinaryLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=51,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 2)  
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        return self.classifier(hn[-1])


# ------------------- TRAINING -------------------
def train_model(train_csv, test_csv, num_epochs=40, batch_size=8, lr=0.0005):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    run_name = f"2class_fall_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    train_dataset = PoseDataset(train_csv)
    test_dataset = PoseDataset(test_csv)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = FallingBinaryLSTM().to(device)

    # Optional class weighting
    #class_counts = np.bincount(train_dataset.seq_labels)
    #weights = 1.0 / class_counts
    #weights = weights / weights.sum()

    #criterion = nn.CrossEntropyLoss(
    #weight=torch.tensor(weights, dtype=torch.float32).to(device))
    
    # Manual Recall Boost (NonFall = 1.0,Fall    = 2.0 or 3.0)
    weights = torch.tensor([1.0, 1.8]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("Loss Weights:", weights)


    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_acc = 0
    best_preds = None
    best_labels = None

    class_names = ['Non-Falling', 'Falling']

    for epoch in range(1, num_epochs + 1):

        # ---------- TRAIN ----------
        model.train()
        train_loss = 0

        for seqs, labels, lengths in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(seqs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_dataset)

        # ---------- TEST ----------
        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for seqs, labels, lengths in test_loader:
                seqs, labels = seqs.to(device), labels.to(device)

                outputs = model(seqs, lengths)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * labels.size(0)

                _, pred = torch.max(outputs, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_dataset)
        acc = accuracy_score(all_labels, all_preds)

        if acc > best_acc:
            best_acc = acc
            best_preds = all_preds.copy()
            best_labels = all_labels.copy()
            torch.save(model.state_dict(), "BEST_MODEL_Fall_Nonfall_LSTM_10fps.pth")

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy/Test", acc, epoch)

        print(f"Epoch {epoch:02d} | Train {train_loss:.4f} | Test {test_loss:.4f} | Acc {acc:.4f}")

    # ---------- FINAL REPORT ----------
    print("\nBest Accuracy:", best_acc)

    cm = confusion_matrix(best_labels, best_preds, labels=[0, 1])

    print("\nConfusion Matrix")
    print("True → / Pred ↓")
    print("              NonFall   Fall")
    print(f"NonFall       {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"Fall          {cm[1,0]:6d}   {cm[1,1]:6d}")

    print("\nClassification Report:")
    print(classification_report(best_labels, best_preds, target_names=class_names))

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    writer.close()
    return model


# ------------------- MAIN -------------------
if __name__ == "__main__":

    model = train_model(
        train_csv="GMDCSA24_10fps_fall_nonfall_train_augmented_ratio.csv",
        test_csv="GMDCSA24_10fps_fall_nonfall_test.csv",
        num_epochs=40,
        batch_size=8,
        lr=0.0005
    )
