import os
import random
import shutil
from pathlib import Path

# === CONFIG ===
SOURCE_DIR = "../DroneVisionDataset/drone_dataset_yolo/dataset_txt"  # your original flat folder with .jpg and .txt
DEST_DIR = "drone_dataset"
VAL_SPLIT = 0.2  # 20% for validation

# === SETUP ===
random.seed(42)
img_exts = ['.jpg', '.jpeg', '.png']

# Create target directories
for subset in ['train', 'val']:
    os.makedirs(f"{DEST_DIR}/images/{subset}", exist_ok=True)
    os.makedirs(f"{DEST_DIR}/labels/{subset}", exist_ok=True)

# Gather all image files
images = [f for f in os.listdir(SOURCE_DIR) if Path(f).suffix.lower() in img_exts]
random.shuffle(images)

val_size = int(len(images) * VAL_SPLIT)

for idx, img_file in enumerate(images):
    subset = 'val' if idx < val_size else 'train'
    label_file = Path(img_file).with_suffix('.txt')

    src_img = os.path.join(SOURCE_DIR, img_file)
    src_lbl = os.path.join(SOURCE_DIR, label_file)

    dst_img = os.path.join(DEST_DIR, 'images', subset, img_file)
    dst_lbl = os.path.join(DEST_DIR, 'labels', subset, label_file)

    shutil.copy2(src_img, dst_img)

    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)
    else:
        print(f"Warning: No label for {img_file}")

print(f"✅ Dataset organized into {DEST_DIR}/images and {DEST_DIR}/labels.")
