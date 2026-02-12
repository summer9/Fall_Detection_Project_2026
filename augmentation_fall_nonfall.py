# FIXED RATIO AUGMENTATION – 2 classes

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ================== CONFIG ==================

INPUT_CSV  = "GMDCSA24_10fps_fall_nonfall_train.csv"
OUTPUT_CSV = "GMDCSA24_10fps_fall_nonfall_train_augmented3_ratio.csv"

COPIES_FALLING     = 3  
COPIES_NONFALLING  = 1 

NOISE_STD = 4.0

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
df['sequence_id'] = df['video_path'] + "_seq_" + df['sequence_id'].astype(str)

print("Sequences created.")

# Separate sequences by class
seq_class = df.groupby('sequence_id')['final_class'].first()

fall_seqs       = seq_class[seq_class=='Falling'].index.tolist()
nonfall_seqs    = seq_class[seq_class=='Non-Falling'].index.tolist()

print(f"Sequences → Falling: {len(fall_seqs)} | Non-Falling: {len(nonfall_seqs)}")

orig_fall       = df[df['final_class'] == 'Falling']['video_path'].nunique()
orig_nonfall    = df[df['final_class'] == 'Non-Falling']['video_path'].nunique()

print(f"Original → Falling: {orig_fall} | Non-Falling: {orig_nonfall}")

print("Applying fixed-ratio augmentation...")
all_dfs = [df.copy()]  # start with original data

for seq_id, group in tqdm(df.groupby('sequence_id'), desc="Augmenting"):
    label = group['final_class'].iloc[0]

    if label == 'Falling':
        n_copies = COPIES_FALLING
    else:  # Non-Falling
        n_copies = COPIES_NONFALLING

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

# Reorder columns so final_class is after "second" (if exists)
cols = final_df.columns.tolist()
if 'final_class' in cols and 'second' in cols:
    cols.remove('final_class')
    second_index = cols.index('second')
    cols.insert(second_index + 1, 'final_class')
    final_df = final_df[cols]

final_df.to_csv(OUTPUT_CSV, index=False)

# ================== FINAL STATS ==================
fall_aug      = final_df[final_df['final_class'] == 'Falling']['video_path'].nunique()
nonfall_aug   = final_df[final_df['final_class'] == 'Non-Falling']['video_path'].nunique()

print("\n" + "="*70)
print("AUGMENTATION DONE")
print(f"Falling      : {fall_aug}")
print(f"Non-Falling  : {nonfall_aug}")
print(f"Total        : {fall_aug + nonfall_aug}")
print("="*70)
