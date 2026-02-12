# assigns a video-level class based on the true classes that actually appear in the frames, 
# using maximally long consecutive runs
# for training LSTM, not use window or stride: did experiments with window and stride but the result is not good.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------- CONFIG --------------------------
csv_file = "final_3class_a.csv"

print("Loading CSV...\n")
df = pd.read_csv(csv_file)
# Verify pose columns
pose_cols = [col for col in df.columns if col.endswith(('_x', '_y', '_c'))]
assert len(pose_cols) == 51, f"Expected 51 pose columns, got {len(pose_cols)}"

# Clean column for safety
df['final_class'] = df['final_class'].astype(str).str.strip()

# -------------------------- Group by video and assign class --------------------------
print("Grouping by video_path and assigning class ...\n")

video_info = []

for video_path, group in df.groupby('video_path'):
    num_frames = len(group)
    if num_frames == 0:
        continue
    
    unique_classes = set(group['final_class'])
    
    has_sleep = 'Sleeping' in unique_classes
    has_fall  = 'Falling'  in unique_classes
    
    if has_sleep:
        label = 0
        class_name = 'Sleeping'
    elif has_fall:
        label = 1
        class_name = 'Falling'
    else:
        label = 2
        class_name = 'Other'
    
    # Optional: store dominant for debugging
    dominant = group['final_class'].value_counts().index[0]
    
    video_info.append({
        'video_path': video_path,
        'num_frames': num_frames,
        'dominant_class': dominant,
        'assigned_class': class_name,
        'label': label,
        'has_sleep': has_sleep,
        'has_fall': has_fall
    })

video_df = pd.DataFrame(video_info)

# -------------------------- Stats & Plot --------------------------
sleeping_lengths = video_df[video_df['label'] == 0]['num_frames']
falling_lengths  = video_df[video_df['label'] == 1]['num_frames']
other_lengths    = video_df[video_df['label'] == 2]['num_frames']

def print_stats(name, data, label_value):
    if len(data) == 0:
        print(f"{name}: No videos")
        return
    print(f"{name} ({label_value}) - {len(data)} videos")
    print(f"  Min:    {data.min():6d} frames (~{data.min()/10:5.1f} s)")
    print(f"  Max:    {data.max():6d} frames (~{data.max()/10:5.1f} s)")
    print(f"  Mean:   {data.mean():6.1f} frames (~{data.mean()/10:5.1f} s)")
    print(f"  Median: {np.median(data):6.0f} frames (~{np.median(data)/10:5.1f} s)")
    print()

print("=" * 80)
print(f"Total videos: {len(video_df):3d}")
print(f"  Sleeping : {len(sleeping_lengths):3d}  ({len(sleeping_lengths)/len(video_df)*100:4.1f}%)")
print(f"  Falling  : {len(falling_lengths):3d}  ({len(falling_lengths)/len(video_df)*100:4.1f}%)")
print(f"  Other    : {len(other_lengths):3d}  ({len(other_lengths)/len(video_df)*100:4.1f}%)")
print("=" * 80)

print_stats("Sleeping", sleeping_lengths, 0)
print_stats("Falling",  falling_lengths,  1)
print_stats("Other",     other_lengths,    2)

# Plot
plt.figure(figsize=(14, 7))
plt.hist(sleeping_lengths, bins=40, alpha=0.6, label='Sleeping (0)', color='dodgerblue', edgecolor='navy')
plt.hist(falling_lengths,  bins=40, alpha=0.6, label='Falling (1)',  color='salmon',     edgecolor='darkred')
plt.hist(other_lengths,    bins=40, alpha=0.5, label='Other (2)',     color='lightgray', edgecolor='gray')

if len(sleeping_lengths) > 0:
    plt.axvline(np.median(sleeping_lengths), color='blue', linestyle='--', linewidth=2,
                label=f"Sleeping median: {np.median(sleeping_lengths):.0f}")
if len(falling_lengths) > 0:
    plt.axvline(np.median(falling_lengths), color='red', linestyle='--', linewidth=2,
                label=f"Falling median: {np.median(falling_lengths):.0f}")
if len(other_lengths) > 0:
    plt.axvline(np.median(other_lengths), color='gray', linestyle='--', linewidth=2,
                label=f"Other median: {np.median(other_lengths):.0f}")

plt.xlabel('Sequence Length (frames @ 10 fps)', fontsize=14)
plt.ylabel('Number of Videos', fontsize=14)
plt.title('Sequence Length Distribution by Class', fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()
