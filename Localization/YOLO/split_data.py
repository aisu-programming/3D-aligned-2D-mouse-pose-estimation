import os
import random
import shutil

"""
Dataset structure before splitting:
dataset/
├── images/
│   ├── MARS_front_00000.jpg
│   ├── MARS_top_00000.jpg
│   ├── ...
├── labels/
│   ├── MARS_front_00000.txt
│   ├── MARS_top_00000.txt
│   ├── ...

After splitting:
dataset/
├── images/
│   ├── train/
│   │   ├── MARS_front_00000.jpg
│   │   ├── MARS_top_00000.jpg
│   │   ├── ...
│   ├── val/
│       ├── MARS_front_00002.jpg
│       ├── MARS_top_00002.jpg
│       ├── ...
├── labels/
│   ├── train/
│   │   ├── MARS_front_00000.txt
│   │   ├── MARS_top_00000.txt
│   │   ├── ...
│   ├── val/
│       ├── MARS_front_00002.txt
│       ├── MARS_top_00002.txt
│       ├── ...
"""

# Paths
BASE_PATH = "Localization/YOLO/"
image_dir = f"{BASE_PATH}dataset/images"
label_dir = f"{BASE_PATH}dataset/labels"
train_image_dir = f"{BASE_PATH}dataset/images/train"
val_image_dir = f"{BASE_PATH}dataset/images/val"
train_label_dir = f"{BASE_PATH}dataset/labels/train"
val_label_dir = f"{BASE_PATH}dataset/labels/val"

# Create directories
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get all front image files
front_images = [f for f in os.listdir(image_dir) if f.startswith("MARS_front") and f.endswith(".jpg")]

# Pair front and top images by their numeric suffix
paired_images = []
for front_image in front_images:
    suffix = front_image.replace("MARS_front_", "").replace(".jpg", "")
    top_image = f"MARS_top_{suffix}.jpg"
    if os.path.exists(os.path.join(image_dir, top_image)):
        paired_images.append((front_image, top_image))

# Shuffle and split the dataset
random.seed(42)  # For reproducibility
random.shuffle(paired_images)
split_ratio = 0.8  # 80% training, 20% validation
split_index = int(len(paired_images) * split_ratio)
train_pairs = paired_images[:split_index]
val_pairs = paired_images[split_index:]


# Move files to train and val folders
def move_files(image_pairs, src_image_dir, src_label_dir, dest_image_dir, dest_label_dir):
    for front_image, top_image in image_pairs:
        # Move front image and its label
        shutil.move(os.path.join(src_image_dir, front_image), os.path.join(dest_image_dir, front_image))
        front_label_file = os.path.splitext(front_image)[0] + ".txt"
        shutil.move(os.path.join(src_label_dir, front_label_file), os.path.join(dest_label_dir, front_label_file))

        # Move top image and its label
        shutil.move(os.path.join(src_image_dir, top_image), os.path.join(dest_image_dir, top_image))
        top_label_file = os.path.splitext(top_image)[0] + ".txt"
        shutil.move(os.path.join(src_label_dir, top_label_file), os.path.join(dest_label_dir, top_label_file))


# Move train files
move_files(train_pairs, image_dir, label_dir, train_image_dir, train_label_dir)

# Move val files
move_files(val_pairs, image_dir, label_dir, val_image_dir, val_label_dir)

print(f"Training pairs: {len(train_pairs)}")
print(f"Validation pairs: {len(val_pairs)}")
