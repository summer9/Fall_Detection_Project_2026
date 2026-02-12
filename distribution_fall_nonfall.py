import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------- CONFIG --------------------------
csv_file = "fall_nonfall_a.csv"
MIN_SEGMENT_LENGTH = 5          # discard segments shorter than this

print("Loading CSV...\n")
df = pd.read_csv(csv_file)

# Verify pose columns
pose_cols = [col for col in df.columns if col.endswith(('_x', '_y', '_c'))]
assert len(pose_cols) == 51, f"Expected 51 pose columns, got {len(pose_cols)}"

# Clean column for safety
df['final_class'] = df['final_class'].astype(str).str.strip()

# Map classes to labels
class_to_label = {'Falling': 0, 'Non-Falling': 1}

# -------------------------- Extract consecutive segments --------------------------
print("Extracting consecutive uniform segments ...\n")

segment_info = []

for video_path, group in df.groupby('video_path'):
    if len(group) == 0:
        continue
    
    labels = group['final_class'].values
    indices = group.index.values  # keep original row indices if needed
    
    current_class = labels[0]
    current_start = 0
    
    for i in range(1, len(labels)):
        if labels[i] != current_class:
            length = i - current_start
            if length >= MIN_SEGMENT_LENGTH:
                segment_info.append({
                    'video_path': video_path,
                    'segment_start_frame': current_start,
                    'segment_end_frame': i - 1,
                    'segment_length': length,
                    'class_name': current_class,
                    'label': class_to_label.get(current_class, 2),
                    'approx_seconds': length / 10.0,
                })
            # Start new segment
            current_class = labels[i]
            current_start = i
    
    # Last segment
    length = len(labels) - current_start
    if length >= MIN_SEGMENT_LENGTH:
        segment_info.append({
            'video_path': video_path,
            'segment_start_frame': current_start,
            'segment_end_frame': len(labels) - 1,
            'segment_length': length,
            'class_name': current_class,
            'label': class_to_label.get(current_class, 2),
            'approx_seconds': length / 10.0,
        })

segment_df = pd.DataFrame(segment_info)

if segment_df.empty:
    print("No segments longer than or equal to {} frames found.".format(MIN_SEGMENT_LENGTH))
    exit()

# -------------------------- Stats & Plot --------------------------
falling_segments = segment_df[segment_df['label'] == 0]
nonfalling_segments  = segment_df[segment_df['label'] == 1]

falling_lengths = falling_segments['segment_length']
nonfalling_lengths  = nonfalling_segments['segment_length']


def print_stats(name, lengths, label_value):
    if len(lengths) == 0:
        print(f"{name} ({label_value}): No segments")
        return
    
    count = len(lengths)
    total_frames = lengths.sum()
    print(f"{name} ({label_value}) - {count} segments")
    print(f"  Total frames in class: {total_frames:,d}")
    print(f"  Min length:    {lengths.min():6d} frames (~{lengths.min()/10:5.1f} s)")
    print(f"  Max length:    {lengths.max():6d} frames (~{lengths.max()/10:5.1f} s)")
    print(f"  Mean length:   {lengths.mean():6.1f} frames (~{lengths.mean()/10:5.1f} s)")
    print(f"  Median length: {np.median(lengths):6.0f} frames (~{np.median(lengths)/10:5.1f} s)")
    print()

print("=" * 80)
print(f"Total segments: {len(segment_df):,d}")
print(f"  falling : {len(falling_lengths):6,d}  ({len(falling_lengths)/len(segment_df)*100:5.1f}%)")
print(f"  nonfallling  : {len(nonfalling_lengths):6,d}  ({len(nonfalling_lengths)/len(segment_df)*100:5.1f}%)")

print(f"  (min segment length = {MIN_SEGMENT_LENGTH} frames)")
print("=" * 80)

print_stats("Falling", falling_lengths, 0)
print_stats("Non-Falling", nonfalling_lengths, 1)



# Optional: show some very long / very short examples
print("\nLongest segments (top 5 per class):")
for label, name in [(0, 'Falling'), (1, 'Non-Falling')]:
    top = segment_df[segment_df['label'] == label].nlargest(5, 'segment_length')
    if not top.empty:
        print(f"\n{name}:")
        print(top[['video_path', 'segment_length', 'approx_seconds', 'segment_start_frame']].to_string(index=False))

# -------------------------- Plot --------------------------
plt.figure(figsize=(14, 7))
plt.hist(falling_lengths, bins=50, alpha=0.6, label='Falling (0)', color='dodgerblue', edgecolor='navy')
plt.hist(nonfalling_lengths,  bins=50, alpha=0.6, label='Non-Falling (1)',  color='salmon', edgecolor='darkred')
 


if len(falling_lengths) > 0:
    plt.axvline(np.median(falling_lengths), color='blue', linestyle='--', linewidth=2,
                label=f"Falling median: {np.median(falling_lengths):.0f}")
if len(nonfalling_lengths) > 0:
    plt.axvline(np.median(nonfalling_lengths), color='red', linestyle='--', linewidth=2,
                label=f"Non-Falling median: {np.median(nonfalling_lengths):.0f}")



plt.xlabel('Segment Length (frames @ 10 fps)', fontsize=14)
plt.ylabel('Number of Segments', fontsize=14)
plt.title('Sequence Length Distribution by Class', fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()