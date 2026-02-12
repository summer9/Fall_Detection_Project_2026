# use dataset with 10fps instead of original 29fps
# split dataset into train, test first. Then add augmentation for training set with balanced ratio sleeping and falling
# untouch test dataset
# this file uses softmax to see exactly probability of 2 classes when model predict falling and testing


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import random
import os
import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- REPRODUCIBILITY -------------------
#torch.manual_seed(42)
#torch.cuda.manual_seed_all(42)
#np.random.seed(42)
#random.seed(42)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# ------------------- DATASET -------------------
class PoseDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        pose_cols = [col for col in df.columns if col.endswith(('_x', '_y', '_c'))]
        assert len(pose_cols) == 51

        self.features = df[pose_cols].values.astype(np.float32)
        class_map = {'Sleeping': 0, 'Falling': 1}
        self.labels = df['class'].map(class_map).values

        self.sequences = []
        self.seq_labels = []
        self.lengths = []

        for video_path in df['video_path'].unique():
            mask = df['video_path'] == video_path
            seq = self.features[mask]
            lab = self.labels[mask][0]
            self.sequences.append(torch.tensor(seq))
            self.seq_labels.append(lab)
            self.lengths.append(len(seq))

        print(f"Created {len(self.sequences)} clean sequences using video_path")
        print("Unique video_path count:", df['video_path'].nunique())


    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.seq_labels[idx], self.lengths[idx]

def collate_fn(batch):
    seqs, labels, lengths = zip(*batch)
    seqs = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return seqs, labels, lengths

# ------------------- MODEL -------------------
class SleepingFallingLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(51, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        return self.classifier(hn[-1])

# ------------------- TRAINING  -------------------
def train_model(train_csv, test_csv, num_epochs=40, batch_size=8, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # TensorBoard
    run_name = f"fall_detection_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    print(f"TensorBoard → http://localhost:6006")
    print(f"Run: tensorboard --logdir=runs\n")

    train_dataset = PoseDataset(train_csv)
    test_dataset  = PoseDataset(test_csv)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = SleepingFallingLSTM().to(device)
    #criterion = nn.CrossEntropyLoss()

    # Manual Recall Boost (Sleeping = 1.0,Falling  = 2.0 or 3.0)
    weights = torch.tensor([1.18, 1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("Loss Weights:", weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_acc = 0.0
    best_preds = None
    best_labels = None

    print(f"Training videos: {len(train_dataset)} | Test videos: {len(test_dataset)}\n")

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for seqs, labels, lengths in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        avg_train_loss = train_loss / len(train_dataset)

        # Validation
        model.eval()
        test_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seqs, labels, lengths in test_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs, lengths)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                _, pred = torch.max(outputs, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss = test_loss / len(test_dataset)
        acc = accuracy_score(all_labels, all_preds)

        # Save best model by accuracy
        if acc > best_acc:
            best_acc = acc
            best_preds = all_preds.copy()
            best_labels = all_labels.copy()
            torch.save(model.state_dict(), 'BEST_MODEL_Sleeping_Falling_LSTM_10fps.pth')

        # TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Test',  avg_test_loss, epoch)
        writer.add_scalar('Accuracy/Test', acc, epoch)
        writer.add_scalar('Accuracy/Best', best_acc, epoch)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {acc:.4f} → Best: {best_acc:.4f}")

    # FINAL RESULTS
    print("\n" + "="*70)
    print("TRAINING FINISHED!")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print("\nCONFUSION MATRIX (Best Model):")
    cm = confusion_matrix(best_labels, best_preds)
    print("                Predicted →")
    print("                  Sleeping   Falling")
    print(f"True Sleeping       {cm[0,0]:6d}      {cm[0,1]:6d}")
    print(f"True Falling        {cm[1,0]:6d}      {cm[1,1]:6d}")
    print("\n" + classification_report(best_labels, best_preds,
                                        target_names=['Sleeping', 'Falling'], digits=4))
    print("Best model saved → BEST_MODEL_BY_ACCURACY_LSTM_10fps.pth")
    print("Open TensorBoard: http://localhost:6006")
    print("="*70)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sleeping', 'Falling'],
                yticklabels=['Sleeping', 'Falling'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title(f'Best Model Confusion Matrix\nAccuracy: {best_acc:.4f}', fontsize=16)
    plt.tight_layout()
    plt.show()
    writer.close()
    return model
# =============================================================================
# PROBABILITY ANALYSIS — SEE EXACTLY HOW THE MODEL DECIDES
# =============================================================================
# ------------------- RUN -------------------
if __name__ == "__main__":
    model = train_model(
        train_csv="GMDCSA24_10fps_train_sleeping_falling_augmented.csv",
        test_csv="GMDCSA24_10fps_test_sleeping_falling.csv",
        num_epochs=40,
        batch_size=8,
        lr=0.001
    )
    print("\n" + "=" * 100)
    print("STARTING DETAILED PROBABILITY ANALYSIS ON TEST SET")
    print("=" * 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Reload best model
    model.load_state_dict(torch.load('BEST_MODEL_Sleeping_Falling_LSTM_10fps.pth'))

    test_dataset = PoseDataset("GMDCSA24_10fps_test_sleeping_falling.csv")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    results = []
    with torch.no_grad():
        for idx, (seq, label, length) in enumerate(test_loader):
            seq = seq.to(device)
            length = length.cpu()

            output = model(seq, length)  # raw logits
            prob = torch.softmax(output, dim=1)[0]  # [prob_sleeping, prob_falling]
            prob_falling = prob[1].item()
            prob_sleeping = prob[0].item()

            pred_05 = "Falling" if prob_falling >= 0.5 else "Sleeping"
            true_class = "Falling" if label.item() == 1 else "Sleeping"
            correct = (pred_05 == true_class)

            results.append({
                'idx': idx,
                'video_path': test_dataset.sequences[idx].shape[0],  # rough identifier
                'true': true_class,
                'pred_0.5': pred_05,
                'prob_falling': round(prob_falling, 4),
                'prob_sleeping': round(prob_sleeping, 4),
                'confidence': round(max(prob_falling, prob_sleeping), 4),
                'correct': correct,
                'error': '' if correct else ('False Alarm' if true_class == 'Sleeping' else 'Missed Fall')
            })

    df = pd.DataFrame(results)
    pd.set_option('display.max_rows', None)
    print(df[[
        'idx', 'true', 'pred_0.5', 'prob_falling', 'prob_sleeping', 'confidence', 'correct', 'error'
    ]].to_string(index=False))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF CONFUSIONS")
    errors = df[~df['correct']]
    if len(errors) == 0:
        print("PREDICTION ON TEST SET!")
    else:
        print(f"{len(errors)} errors:")
        print(errors[['idx', 'true', 'pred_0.5', 'prob_falling', 'error']].to_string(index=False))



