import os
import json
import shutil

""" 
YOLO ANNOTATION FORMAT
<class_id> <x_center> <y_center> <width> <height>

dataset/
├── images/
│   ├── MARS_front_00000.jpg
│   ├── MARS_front_00001.jpg
├── labels/
│   ├── MARS_front_00000.txt
│   ├── MARS_front_00001.txt
"""
# Constants
MARGIN = 25
CLASS_MAPPING = {"black": 0, "white": 1}  # Map "black" to 0, "white" to 1
TOP = "top"
FRONT = "front"

IMG_WIDTH = {TOP: 1024, FRONT:1280}
IMG_HEIGHT = {TOP: 570, FRONT: 500}

# Function to compute bounding box
def compute_bounding_box(coords, margin, img_width, img_height):
    x_min = max(0, min(coords["x"]) - margin)
    y_min = max(0, min(coords["y"]) - margin)
    x_max = min(img_width, max(coords["x"]) + margin)
    y_max = min(img_height, max(coords["y"]) + margin)

    # Convert to YOLO format
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return x_center, y_center, width, height


# 1. Output directories
os.makedirs("Localization/YOLO/dataset/labels/train", exist_ok=True)
os.makedirs("Localization/YOLO/dataset/images/train", exist_ok=True)


json_paths = {TOP: "datasets/MARS/MARS_keypoints_top.json", FRONT: "datasets/MARS/MARS_front_raw.json"}
for view_type in [TOP, FRONT]:
    # 2. Load JSON data
    json_path = json_paths[view_type]
    with open(json_path, "r") as f:
        dataset = json.load(f)

    # 3. Parse the JSON File
    # Read the JSON file to extract the coordinates for "black" and "white" mice.
    for entry in dataset:
        image_name = entry["filename"]
        coords = entry["coords"]

        # Initialize annotation lines
        annotations = []

        # Process "black" mouse
        if "black" in coords:
            bbox = compute_bounding_box(coords["black"], MARGIN, IMG_WIDTH[view_type], IMG_HEIGHT[view_type])
            annotations.append(f"{CLASS_MAPPING['black']} {' '.join(map(str, bbox))}")

        # Process "white" mouse
        if "white" in coords:
            bbox = compute_bounding_box(coords["white"], MARGIN, IMG_WIDTH[view_type], IMG_HEIGHT[view_type])
            annotations.append(f"{CLASS_MAPPING['white']} {' '.join(map(str, bbox))}")

        # Save annotation file
        label_path = f"Localization/YOLO/dataset/labels/{os.path.splitext(image_name)[0]}.txt"
        # Ensure the parent directory for the label file exists
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w") as label_file:
            label_file.write("\n".join(annotations))
        # Copy the image to the destination instead of moving (if needed)
        os.makedirs(os.path.dirname(f"Localization/YOLO/dataset/images/{image_name}"), exist_ok=True)
        original_path = {TOP: "raw_images_top", FRONT: "raw_images_front"}
        shutil.copy(f"datasets/MARS/{original_path[view_type]}/{image_name}", f"Localization/YOLO/dataset/images/{image_name}")
