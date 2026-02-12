
# SIMPLE NOISE AUGMENTATION
# Resolution of video  1280x720
# Add augmentation to training set, balance 2 classes: sleeping and falling, just add to x, y, not c


import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ================== CONFIGURATION ==================

input_csv = "GMDCSA24_10fps_train_sleeping_falling.csv"
output_csv = "GMDCSA24_10fps_train_sleeping_falling_augmented.csv"

# How many noisy versions to create
COPIES_SLEEPING = 13 #
COPIES_FALLING = 1  #

# noise level for 1280x720 + YOLOv11-pose
NOISE_STD = 4.0
# this is Guassian noise
SEED = 42
np.random.seed(SEED)

# ===================================================

print("Loading original dataset...")
df = pd.read_csv(input_csv)

print(f"Original → Sleeping: {df[df['class'] == 'Sleeping']['video_path'].nunique()} | "
      f"Falling: {df[df['class'] == 'Falling']['video_path'].nunique()}")

all_dfs = [df.copy()]  # Keep original data
counter = 0

print("Adding augmentation YOLOv11-pose noise...")
for video_path, group in tqdm(df.groupby('video_path'), desc="Augmenting"):
    label = group['class'].iloc[0]
    n_copies = COPIES_SLEEPING if label == 'Sleeping' else COPIES_FALLING

    # Extract clean base name (S1/ADL/03.mp4)
    base_path = video_path
    for i in range(n_copies):
        aug = group.copy()

        # Add realistic noise
        coord_cols = [c for c in aug.columns if c.endswith(('_x', '_y'))]
        noise = np.random.normal(0, NOISE_STD, size=aug[coord_cols].shape)
        aug[coord_cols] += noise


        noise_suffix = f"_noise_{i + 1:03d}"  # _noise_001, _noise_002, ...
        new_video_path = base_path.replace('.mp4', f'{noise_suffix}.mp4')
        new_video_name = os.path.basename(new_video_path).replace('.mp4', '')

        aug['video_path'] = new_video_path
        aug['video_name'] = new_video_name
        aug['frame'] = np.arange(len(aug))  # reset frame counter

        all_dfs.append(aug)

# Combine everything
final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv(output_csv, index=False)

print("\n" + "="*60)
print("AUGMENTATION DONE!")
print(f"Output → {output_csv}")
print(f"Example names in new CSV:")
example_sleep = final_df[final_df['class']=='Sleeping']['video_path'].unique()[:3]
example_fall  = final_df[final_df['class']=='Falling']['video_path'].unique()[:3]
print("  Sleeping →", example_sleep)
print("  Falling  →", example_fall)
print(f"Total sequences → {final_df['video_path'].nunique()}")
print("="*60)

sleeping_count = final_df[final_df['class'] == 'Sleeping']['video_path'].nunique()
falling_count = final_df[final_df['class'] == 'Falling']['video_path'].nunique()
total_count = final_df['video_path'].nunique()

print(f"After augmentation:")
print(f"  Sleeping sequences: {sleeping_count}")
print(f"  Falling sequences : {falling_count}")
print(f"  Total sequences   : {total_count}")
print(f"  Balance ratio (Sleeping:Falling) = {sleeping_count}:{falling_count} ≈ {sleeping_count/falling_count:.2f}:1")
