
# FIXED RATIO AUGMENTATION – 3 classes


import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ================== CONFIG ==================

INPUT_CSV  = "GMDCSA24_10fps_3classes_a_train.csv"
OUTPUT_CSV = "GMDCSA24_10fps_3classes_a_train_augmented3_ratio.csv"

COPIES_SLEEPING = 6   # 6 new + 1 original = 7
COPIES_FALLING  = 1   # 1 new + 1 original = 2
COPIES_OTHER    = 0   # unchanged

NOISE_STD = 3.0

IMG_WIDTH  = 1280
IMG_HEIGHT = 720
CLIP_COORDINATES = True

SEED = 42
np.random.seed(SEED)

# =============================================

print("Loading dataset...")
df = pd.read_csv(INPUT_CSV)


print("Creating TRUE sequence IDs...")

df = df.sort_values(['video_path','frame'])

# Detect frame gaps OR class changes
df['frame_gap'] = df.groupby('video_path')['frame'].diff().fillna(1) != 1
df['class_change'] = df.groupby('video_path')['final_class'].shift() != df['final_class']

df['new_sequence'] = (df['frame_gap'] | df['class_change']).astype(int)

df['sequence_id'] = df.groupby('video_path')['new_sequence'].cumsum()

# Combine video_path + sequence_id → unique sequence
df['sequence_id'] = df['video_path'] + "_seq_" + df['sequence_id'].astype(str)

print("Sequences created.")

seq_class = df.groupby('sequence_id')['final_class'].first()

sleep_seqs = seq_class[seq_class=='Sleeping'].index.tolist()
fall_seqs  = seq_class[seq_class=='Falling'].index.tolist()
other_seqs = seq_class[seq_class=='Other'].index.tolist()

print(f"Sequences → Sleeping: {len(sleep_seqs)} | Falling: {len(fall_seqs)} | Other: {len(other_seqs)}")


orig_sleep = df[df['final_class'] == 'Sleeping']['video_path'].nunique()
orig_fall  = df[df['final_class'] == 'Falling']['video_path'].nunique()
orig_other = df[df['final_class'] == 'Other']['video_path'].nunique()

print(f"Original → Sleeping: {orig_sleep} | Falling: {orig_fall} | Other: {orig_other}")


print("Applying fixed-ratio augmentation...")
all_dfs = [df.copy()]  # start with original data

for seq_id, group in tqdm(df.groupby('sequence_id'), desc="Augmenting"):
    label = group['final_class'].iloc[0]

    if label == 'Sleeping':
        n_copies = COPIES_SLEEPING
    elif label == 'Falling':
        n_copies = COPIES_FALLING
    else:
        n_copies = COPIES_OTHER
    # Apply augmentation
    for i in range(n_copies):
        aug = group.copy()

        # Add Gaussian noise to coordinate columns
        coord_cols = [c for c in aug.columns if c.endswith(('_x', '_y'))]
        noise = np.random.normal(0, NOISE_STD, size=aug[coord_cols].shape)
        aug[coord_cols] += noise

        # Clip coordinates to image boundaries
        if CLIP_COORDINATES:
            for c in coord_cols:
                if '_x' in c:
                    aug[c] = aug[c].clip(0, IMG_WIDTH)
                else:
                    aug[c] = aug[c].clip(0, IMG_HEIGHT)

        # Update video path & name with suffix
        orig_video_path = group['video_path'].iloc[0]
        suffix = f"_aug{i+1:03d}"
        new_video_path = orig_video_path.replace('.mp4', f'{suffix}.mp4')
        new_video_name = os.path.basename(new_video_path).replace('.mp4', '')

        aug['video_path'] = new_video_path
        aug['video_name'] = new_video_name
        aug['frame'] = np.arange(len(aug))  # reset frame numbers

        all_dfs.append(aug)
        

# Combine and save
final_df = pd.concat(all_dfs, ignore_index=True)

# ===== Reorder columns so final_class is after "second" =====

cols = final_df.columns.tolist()

if 'final_class' in cols and 'second' in cols:
    cols.remove('final_class')
    second_index = cols.index('second')
    cols.insert(second_index + 1, 'final_class')
    final_df = final_df[cols]

# Save
final_df.to_csv(OUTPUT_CSV, index=False)


# ================== FINAL STATS ==================

sleep_aug = final_df[final_df['final_class'] == 'Sleeping']['video_path'].nunique()
fall_aug  = final_df[final_df['final_class'] == 'Falling']['video_path'].nunique()
other_aug = final_df[final_df['final_class'] == 'Other']['video_path'].nunique()

print("\n" + "="*70)
print("AUGMENTATION DONE")
print(f"Sleeping : {sleep_aug}  (expected **105**)")
print(f"Falling  : {fall_aug}   (expected **146**)")
print(f"Other    : {other_aug}  (expected **135**)")
print(f"Total    : {sleep_aug + fall_aug + other_aug}")
print("="*70)
