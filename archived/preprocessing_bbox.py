import os
import json
import cv2
import numpy as np

# Constants
MARGIN = 25
CLASS_MAPPING = {"black": 0, "white": 1}
TOP = "top"
FRONT = "front"

IMG_WIDTH = {TOP: 1024, FRONT: 1280}
IMG_HEIGHT = {TOP: 570, FRONT: 500}

# Function to compute bounding box
def compute_bounding_box(coords, margin, img_width, img_height):
    x_min = int(max(0, min(coords["x"]) - margin))
    y_min = int(max(0, min(coords["y"]) - margin))
    x_max = int(min(img_width, max(coords["x"]) + margin))
    y_max = int(min(img_height, max(coords["y"]) + margin))
    return x_min, y_min, x_max, y_max

# Function to paste cropped image onto a black canvas
def paste_on_black_canvas(cropped_image, max_width, max_height):
    black_canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    h, w, _ = cropped_image.shape

    # Calculate center placement
    y_offset = (max_height - h) // 2
    x_offset = (max_width - w) // 2

    # Paste the cropped image onto the black canvas
    black_canvas[y_offset:y_offset + h, x_offset:x_offset + w] = cropped_image
    return black_canvas

# Function to adjust keypoints for placement on the black canvas
def adjust_keypoints(keypoints, x_offset, y_offset):
    adjusted_keypoints = {"x": [], "y": []}
    for x, y in zip(keypoints["x"], keypoints["y"]):
        adjusted_keypoints["x"].append(x + x_offset)
        adjusted_keypoints["y"].append(y + y_offset)
    return adjusted_keypoints

# Function to draw keypoints on an image
def draw_keypoints(image, keypoints, color=(0, 255, 0), radius=3, thickness=-1):
    for x, y in zip(keypoints["x"], keypoints["y"]):
        cv2.circle(image, (int(x), int(y)), radius, color, thickness)


def crop_and_draw(image_path, output_folder, bboxes, class_mapping, max_width, max_height, coords):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image {image_path} not found!")

    # Create a copy of the image to draw bounding boxes
    image_with_bbox = original_image.copy()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    adjusted_keypoints = {}  # To store adjusted keypoints for each class

    for bbox, class_id in bboxes:
        x_min, y_min, x_max, y_max = bbox
        color = (0, 0, 255) if class_id == class_mapping["black"] else (255, 255, 255)
        label = "black" if class_id == class_mapping["black"] else "white"

        # Draw rectangle and label on the image with bounding boxes
        cv2.rectangle(image_with_bbox, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image_with_bbox, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Crop the original image (not the one with bounding boxes)
        cropped_image = original_image[y_min:y_max, x_min:x_max]

        # Paste cropped image onto a black canvas
        padded_image = paste_on_black_canvas(cropped_image, max_width, max_height)

        # Save the canvas-pasted image without keypoints
        cropped_filename_without_keypoints = os.path.join(output_folder, f"{label}.jpg")
        cv2.imwrite(cropped_filename_without_keypoints, padded_image)

        # Calculate offsets for keypoint adjustment
        h, w, _ = cropped_image.shape
        y_offset = (max_height - h) // 2 - y_min
        x_offset = (max_width - w) // 2 - x_min

        # Adjust keypoints for the current class
        adjusted_keypoints[label] = adjust_keypoints(coords[label], x_offset, y_offset)
        
        ''' if you want to also save the cropped images with adjusted keypoints, uncomment these lines
        
        # Draw adjusted keypoints on the black canvas image
        draw_keypoints(padded_image, adjusted_keypoints[label])

        # Save the canvas-pasted image with keypoints
        cropped_filename_with_keypoints = os.path.join(output_folder, f"{label}_with_keypoints.jpg")
        cv2.imwrite(cropped_filename_with_keypoints, padded_image)
        
        '''

    # Save the adjusted keypoints
    keypoints_filename = os.path.join(output_folder, "adjusted_keypoints.json")
    with open(keypoints_filename, "w") as kp_file:
        json.dump(adjusted_keypoints, kp_file, indent=4)

    # Save the image with bounding boxes
    output_image_path = os.path.join(output_folder, f"{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, image_with_bbox)

    
# Process each dataset
json_paths = {TOP: "MARS_keypoints_top.json", FRONT: "MARS_keypoints_front.json"}
original_path = {TOP: "raw_images_top", FRONT: "raw_images_front"}

for view_type in [TOP, FRONT]:
    json_path = json_paths[view_type]
    with open(json_path, "r") as f:
        dataset = json.load(f)

    total_files = len(dataset)
    next_progress = 10  # Initialize progress tracking
    max_width, max_height = 0, 0  # Track the maximum dimensions

    # First pass: calculate maximum width and height
    max_width_info = {"image": None, "type": None}
    max_height_info = {"image": None, "type": None}

    for entry in dataset:
        coords = entry["coords"]
        image_name = entry["filename"]
        if "black" in coords:
            x_min, y_min, x_max, y_max = compute_bounding_box(coords["black"], MARGIN, IMG_WIDTH[view_type], IMG_HEIGHT[view_type])
            width = x_max - x_min
            height = y_max - y_min
            if width > max_width:
                max_width = width
                max_width_info = {"image": image_name, "type": "black"}
            if height > max_height:
                max_height = height
                max_height_info = {"image": image_name, "type": "black"}
        if "white" in coords:
            x_min, y_min, x_max, y_max = compute_bounding_box(coords["white"], MARGIN, IMG_WIDTH[view_type], IMG_HEIGHT[view_type])
            width = x_max - x_min
            height = y_max - y_min
            if width > max_width:
                max_width = width
                max_width_info = {"image": image_name, "type": "white"}
            if height > max_height:
                max_height = height
                max_height_info = {"image": image_name, "type": "white"}

    # Print the max width and height details
    print(f"Maximum bounding box width: {max_width}px, found in image: {max_width_info['image']} ({max_width_info['type']})")
    print(f"Maximum bounding box height: {max_height}px, found in image: {max_height_info['image']} ({max_height_info['type']})")

    # Second pass: process images
    for idx, entry in enumerate(dataset, start=1):
        image_name = entry["filename"]
        coords = entry["coords"]

        bboxes = []

        # Process black and white mice
        if "black" in coords:
            bbox = compute_bounding_box(coords["black"], MARGIN, IMG_WIDTH[view_type], IMG_HEIGHT[view_type])
            bboxes.append((bbox, CLASS_MAPPING["black"]))
        if "white" in coords:
            bbox = compute_bounding_box(coords["white"], MARGIN, IMG_WIDTH[view_type], IMG_HEIGHT[view_type])
            bboxes.append((bbox, CLASS_MAPPING["white"]))

        # Define paths
        source_image_path = f"{original_path[view_type]}/{original_path[view_type]}/{image_name}"
        cropped_image_folder = f"Localization/Ground/{os.path.splitext(image_name)[0]}"

        # Crop and save images with adjusted keypoints
        crop_and_draw(source_image_path, cropped_image_folder, bboxes, CLASS_MAPPING, max_width, max_height, coords)

        # Print progress
        progress = (idx / total_files) * 100
        if progress >= next_progress:
            print(f"Processing {view_type} images: {int(progress)}% completed")
            next_progress += 10
