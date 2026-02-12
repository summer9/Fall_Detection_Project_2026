# use dataset with 10fps instead of original 29fps
# split dataset into train, test first. Then add augmentation for training set
# untouch test dataset
# this file uses softmax to see probabilities for 3 classes: Sleeping, Falling, Other

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
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# np.random.seed(42)
# random.seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# ------------------- DATASET -------------------
class PoseDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        pose_cols = [col for col in df.columns if col.endswith(('_x', '_y'))]
        assert len(pose_cols) == 34, f"Expected 51 pose columns, got {len(pose_cols)}"

        self.features = df[pose_cols].values.astype(np.float32)
        
        class_map = {'Sleeping': 0, 'Falling': 1, 'Other': 2}
        labels_series = df['final_class'].map(class_map)
        
        if labels_series.isnull().any():
            unmapped = df['final_class'][labels_series.isnull()].unique()
            raise ValueError(
                f"Unmapped classes found: {list(unmapped)}\n"
                f"Allowed classes: Sleeping, Falling, Other\n"
                f"Check your CSV file: {csv_file}"
            )
        

        self.labels = labels_series.values.astype(np.int64)  # per-frame labels
        
        self.sequences = []
        self.seq_labels = []
        self.lengths = []

        for video_path in df['video_path'].unique():
            mask = df['video_path'] == video_path
            video_features = self.features[mask]
            video_labels = self.labels[mask]
            
            if len(video_features) == 0:
                continue

            # Find consecutive segments with the same label
            current_label = video_labels[0]
            start_idx = 0

            for i in range(1, len(video_labels)):
                if video_labels[i] != current_label:
                    # End of a segment → save it
                    if i - start_idx >= 5:  # optional: skip very short segments (<5 frames)
                        seq = video_features[start_idx:i]
                        self.sequences.append(torch.tensor(seq, dtype=torch.float32))
                        self.seq_labels.append(int(current_label))
                        self.lengths.append(i - start_idx)
                    
                    # Start new segment
                    current_label = video_labels[i]
                    start_idx = i

            # Don't forget the last segment
            if len(video_labels) - start_idx >= 5:
                seq = video_features[start_idx:]
                self.sequences.append(torch.tensor(seq, dtype=torch.float32))
                self.seq_labels.append(int(current_label))
                self.lengths.append(len(video_labels) - start_idx)

        print(f"Created {len(self.sequences)} variable-length segments from {csv_file}")
        print("Class distribution (segments):", np.bincount(self.seq_labels))
        
        if len(self.sequences) == 0:
            raise ValueError("No valid segments found in the dataset!")

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.seq_labels[idx], self.lengths[idx]

def collate_fn(batch):
    seqs, labels, lengths = zip(*batch)
    seqs = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return seqs, labels, lengths


# ------------------- MODEL -------------------
class SleepingFallingOtherLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(34, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 3)          # 3 classes now
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
    run_name = f"3class_fall_detection_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    print(f"TensorBoard → http://localhost:6006")
    print(f"Run: tensorboard --logdir=runs\n")

  
    train_dataset = PoseDataset(train_csv)
    test_dataset  = PoseDataset(test_csv)
    

    print("\n" + "="*50)
    print("CLASS DISTRIBUTION CHECK")
    print("="*50)

    def print_class_dist(dataset, name):
        if not dataset.seq_labels:
            print(f"No sequences in {name} dataset!")
            return
    
        labels_np = np.array(dataset.seq_labels)
        unique, counts = np.unique(labels_np, return_counts=True)
        class_names_local = ['Sleeping', 'Falling', 'Other']
    
        print(f"\n{name} set: {len(dataset)} sequences (videos)")
        for cls, cnt in zip(unique, counts):
            class_name = class_names_local[cls] if cls < len(class_names_local) else f"Class {cls} (unknown)"
            print(f"  {class_name:10} (label {cls}): {cnt} sequences")
    
        # Also show frame-level distribution (optional but helpful)
        all_labels_frames = []
        for i in range(len(dataset)):
            all_labels_frames.extend([dataset.seq_labels[i]] * dataset.lengths[i])
        frame_unique, frame_counts = np.unique(all_labels_frames, return_counts=True)
        print(f"  (frame-level total: {len(all_labels_frames)} frames)")
        for cls, cnt in zip(frame_unique, frame_counts):
            print(f"    → label {cls}: {cnt} frames")

    print_class_dist(train_dataset, "TRAIN")
    print_class_dist(test_dataset,  "TEST")
    print("="*50 + "\n")

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    model = SleepingFallingOtherLSTM().to(device)
    criterion = nn.CrossEntropyLoss()

    #class_counts = np.bincount(train_dataset.seq_labels)
    #weights = 1.0 / class_counts
    #weights = weights / weights.sum()
    #criterion = nn.CrossEntropyLoss(
    #weight=torch.tensor(weights, dtype=torch.float32).to(device))



    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_acc = 0.0
    best_preds = None
    best_labels = None

    class_names = ['Sleeping', 'Falling', 'Other']
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
            torch.save(model.state_dict(), 'BEST_MODEL_3CLASSES_LSTM_10fps.pth')

        # TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Test',  avg_test_loss, epoch)
        writer.add_scalar('Accuracy/Test', acc, epoch)
        writer.add_scalar('Accuracy/Best', best_acc, epoch)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {acc:.4f} → Best: {best_acc:.4f}")

    # FINAL RESULTS
    print("\n" + "="*80)
    print("TRAINING FINISHED!")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print("\nCONFUSION MATRIX (Best Model):")
    cm = confusion_matrix(best_labels, best_preds, labels=[0, 1, 2])
    print("                Predicted →")
    print("True   Sleeping   Falling   Other")
    for i, name in enumerate(class_names):
        print(f"{name:7} {cm[i,0]:8d} {cm[i,1]:8d} {cm[i,2]:6d}")
    
    print("\n" + classification_report(best_labels, best_preds,
                                        target_names=class_names, digits=4))
    print("Best model saved → BEST_MODEL_3CLASSES_LSTM_10fps.pth")
    print("Open TensorBoard: http://localhost:6006")
    print("="*80)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title(f'Best Model Confusion Matrix\nAccuracy: {best_acc:.4f}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    writer.close()
    return model


# =============================================================================
# PROBABILITY ANALYSIS — SEE EXACTLY HOW THE MODEL DECIDES (3 classes)
# =============================================================================
if __name__ == "__main__":
    model = train_model(
        train_csv="GMDCSA24_10fps_3classes_a_train_augmented_ratio.csv",
        test_csv="GMDCSA24_10fps_3classes_a_test.csv",
        num_epochs=40,
        batch_size=8,
        lr=0.001
    )

    print("\n" + "=" * 100)
    print("STARTING DETAILED PROBABILITY ANALYSIS ON TEST SET (3 classes)")
    print("=" * 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Reload best model
    model.load_state_dict(torch.load('BEST_MODEL_3CLASSES_LSTM_10fps.pth'))

    test_dataset = PoseDataset("GMDCSA24_10fps_3classes_a_test.csv")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    class_names = ['Sleeping', 'Falling', 'Other']
    results = []
    with torch.no_grad():
        for idx, (seq, label, length) in enumerate(test_loader):
            seq = seq.to(device)
            length = length.cpu()

            output = model(seq, length)  # raw logits [1, 3]
            prob = torch.softmax(output, dim=1)[0]  # [p_sleep, p_fall, p_other]
            
            probs = [round(p.item(), 4) for p in prob]
            pred_idx = torch.argmax(prob).item()
            pred_class = class_names[pred_idx]
            true_class = class_names[label.item()]

            correct = (pred_idx == label.item())

            results.append({
                'idx': idx,
                'true': true_class,
                'pred': pred_class,
                'p_Sleeping': probs[0],
                'p_Falling': probs[1],
                'p_Other':   probs[2],
                'confidence': round(prob.max().item(), 4),
                'correct': correct,
                'error': '' if correct else f"Wrong → {true_class}"
            })

    df = pd.DataFrame(results)
    pd.set_option('display.max_rows', None)
    print(df[[
        'idx', 'true', 'pred', 'p_Sleeping', 'p_Falling', 'p_Other', 
        'confidence', 'correct', 'error'
    ]].to_string(index=False))

    # Summary of mistakes
    print("\n" + "=" * 80)
    print("SUMMARY OF ERRORS")
    errors = df[~df['correct']]
    if len(errors) == 0:
        print("PERFECT PREDICTION ON TEST SET!")
    else:
        print(f"{len(errors)} errors found:")
        print(errors[['idx', 'true', 'pred', 'p_Sleeping', 'p_Falling', 'p_Other', 'error']].to_string(index=False))