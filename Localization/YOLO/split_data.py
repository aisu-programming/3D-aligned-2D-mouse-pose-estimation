import os
import random
import shutil

"""
Before splitting:
dataset/
├── images/
│   ├── MARS_front_00000.jpg
│   ├── MARS_front_00001.jpg
│   ├── ...
├── labels/
│   ├── MARS_front_00000.txt
│   ├── MARS_front_00001.txt
│   ├── ...

After splitting:
dataset/
├── images/
│   ├── train/
│   │   ├── MARS_front_00000.jpg
│   │   ├── ...
│   ├── val/
│       ├── MARS_front_00002.jpg
│       ├── ...
├── labels/
│   ├── train/
│   │   ├── MARS_front_00000.txt
│   │   ├── ...
│   ├── val/
│       ├── MARS_front_00002.txt
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

# Get all image files
images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Shuffle and split the dataset
random.seed(42)  # For reproducibility
random.shuffle(images)
split_ratio = 0.8  # 80% training, 20% validation
split_index = int(len(images) * split_ratio)
train_images = images[:split_index]
val_images = images[split_index:]


# Move files to train and val folders
def move_files(image_list, src_image_dir, src_label_dir, dest_image_dir, dest_label_dir):
    for image in image_list:
        # Image file
        shutil.move(os.path.join(src_image_dir, image), os.path.join(dest_image_dir, image))

        # Label file (assuming the label file has the same name as the image but with .txt extension)
        label_file = os.path.splitext(image)[0] + ".txt"
        shutil.move(os.path.join(src_label_dir, label_file), os.path.join(dest_label_dir, label_file))


# Move train files
move_files(train_images, image_dir, label_dir, train_image_dir, train_label_dir)

# Move val files
move_files(val_images, image_dir, label_dir, val_image_dir, val_label_dir)

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
