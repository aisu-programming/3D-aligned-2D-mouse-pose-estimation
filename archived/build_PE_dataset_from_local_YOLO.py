import os
import cv2
import json
import numpy as np
from tqdm import tqdm



# Constants
CLASS_MAPPING = {"black": 0, "white": 1}
TOP = "top"
FRONT = "front"

IMG_WIDTH = {TOP: 1024, FRONT: 1280}
IMG_HEIGHT = {TOP: 570, FRONT: 500}

# YOLO Prediction Folder
YOLO_PREDICTION_FOLDER = "Localization/YOLO/predictions/"



# Function to parse YOLO bounding boxes
def parse_yolo_bboxes(yolo_file, img_width, img_height):
    bboxes = []
    if not os.path.exists(yolo_file):
        return bboxes  # Return an empty list if the file doesn't exist

    with open(yolo_file, "r") as f:
        for line in f:
            values = line.strip().split()
            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:5])  # Only unpack the first 4 values
            
            # Convert normalized YOLO bbox to pixel values
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)
            
            bboxes.append(((x_min, y_min, x_max, y_max), class_id))

    bboxes = sorted(bboxes, key=lambda bbox: bbox[1])
    return bboxes

# Function to crop an object from the image using the bounding box
def crop_object_from_image(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    # Ensure the bounding box is valid
    if x_min < x_max and y_min < y_max:
        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image
    return None

# Function to create a black canvas and paste the cropped image onto it
def paste_on_black_canvas(cropped_image, max_width, max_height):
    # Create a black canvas of the maximum width and height
    canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    
    # Get dimensions of the cropped image
    cropped_h, cropped_w = cropped_image.shape[:2]
    
    # Calculate the top-left corner for centering
    start_x = (max_width - cropped_w) // 2
    start_y = (max_height - cropped_h) // 2
    
    # Paste the cropped image onto the canvas
    try:
        canvas[start_y:start_y + cropped_h, start_x:start_x + cropped_w] = cropped_image
    except ValueError as e:
        print(start_x, cropped_w, start_y, cropped_h)
        raise e
    return canvas

# Function to draw bounding boxes and annotate black/white labels
def draw_bounding_boxes(image, bboxes):
    for (x_min, y_min, x_max, y_max), class_id in bboxes:
        # Set the color for the bounding box (green for black, red for white)
        color = (0, 255, 0) if class_id == CLASS_MAPPING["black"] else (255, 0, 0)
        
        # Draw the rectangle (bounding box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Set the label text based on the class_id
        label = "Black" if class_id == CLASS_MAPPING["black"] else "White"
        
        # Set text color (white for readability) and font
        text_color = (255, 255, 255)  # White text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Position the text slightly above the bounding box
        text_x = x_min
        text_y = y_min - 10
        
        # Draw the label text on the image
        cv2.putText(image, label, (text_x, text_y), font, font_scale, text_color, thickness)

    return image

# this is for adjusting
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



# Main loop for processing
json_paths = {TOP: "datasets/MARS/MARS_keypoints_top.json", FRONT: "datasets/MARS/MARS_keypoints_front.json"}
original_path = {TOP: "datasets/MARS/raw_images_top", FRONT: "datasets/MARS/raw_images_front"}

for view_type in [FRONT, TOP]:
    json_path = json_paths[view_type]
    with open(json_path, "r") as f:
        dataset = json.load(f)



    """ Uncomment the below part to calculate the original max width and max height """
    # # Customized max width and max height
    # max_width  = { TOP: 360, FRONT: 480 }[view_type]
    # max_height = { TOP: 360, FRONT: 480 }[view_type]
    print(f"Calculating the max width and height from {view_type} images...")
    max_width = 0
    max_height = 0
    max_width_image = ""
    max_height_image = ""
    max_width_class = ""
    max_height_class = ""
    # Calculate max width and height
    for idx, entry in enumerate(dataset):
        # Construct the YOLO prediction file path
        img_path = f"{original_path[view_type]}/{entry['filename']}"
        yolo_file = os.path.join(YOLO_PREDICTION_FOLDER, f"{os.path.splitext(entry['filename'])[0]}.txt")
        if not os.path.exists(img_path) or not os.path.exists(yolo_file):
            continue
        bboxes = parse_yolo_bboxes(yolo_file, IMG_WIDTH[view_type], IMG_HEIGHT[view_type])
        for (bbox, class_id) in bboxes:
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            if width > max_width:
                max_width = width
                max_width_image = entry["filename"]
                max_width_class = "black" if class_id == CLASS_MAPPING["black"] else "white"
            if height > max_height:
                max_height = height
                max_height_image = entry["filename"]
                max_height_class = "black" if class_id == CLASS_MAPPING["black"] else "white"
    # Log the maximum bounding box details
    print(f"Maximum bounding box width: {max_width}px, found in image: {max_width_image} ({max_width_class})")
    print(f"Maximum bounding box height: {max_height}px, found in image: {max_height_image} ({max_height_class})")
    max_width = max(max_height, max_width)
    max_height = max_width



    adjusted_keypoint_list = []  # To store all adjusted keypoints for each view_type

    # Process images and save cropped images on black canvas and annotated images
    for idx, entry in tqdm(enumerate(dataset), desc=f"Processing {view_type} images", total=len(dataset)):
        
        img_path = f"{original_path[view_type]}/{entry['filename']}"
        yolo_file = os.path.join(YOLO_PREDICTION_FOLDER, f"{os.path.splitext(entry['filename'])[0]}.txt")
        if not os.path.exists(img_path) or not os.path.exists(yolo_file):
            continue

        original_image = cv2.imread(img_path)
        bboxes = parse_yolo_bboxes(yolo_file, IMG_WIDTH[view_type], IMG_HEIGHT[view_type])
        

        """ Uncomment if you to save the visualized results of YOLO localization """
        # # Draw and save the annotated image with the same name as the folder
        # annotated_output_folder = os.path.join("datasets", "MARS", f"raw_images_{view_type}", "YOLO_localized")
        # os.makedirs(annotated_output_folder, exist_ok=True)
        # annotated_image_path = os.path.join(annotated_output_folder, entry['filename'])
        # annotated_image = draw_bounding_boxes(original_image.copy(), bboxes)
        # cv2.imwrite(annotated_image_path, annotated_image)


        cropped_output_folder = os.path.join("datasets", "MARS", f"cropped_images_{view_type}_YOLO")
        os.makedirs(cropped_output_folder, exist_ok=True)

        # Create and save images with black canvas
        for (bbox, class_id) in bboxes:

            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            class_name = ""
            cropped_image = crop_object_from_image(original_image, bbox)

            if cropped_image is not None:
                canvas = paste_on_black_canvas(cropped_image, max_width, max_height)
                
                # Determine filename based on class
                if class_id == CLASS_MAPPING["black"]:
                    class_name="black"
                    cropped_filename = f"{os.path.splitext(entry['filename'])[0]}_black.jpg"
                elif class_id == CLASS_MAPPING["white"]:
                    class_name="white"
                    cropped_filename = f"{os.path.splitext(entry['filename'])[0]}_white.jpg"
                else:
                    continue
                
                cropped_image_path = os.path.join(cropped_output_folder, cropped_filename)
                cv2.imwrite(cropped_image_path, canvas)
                
                # Calculate offsets for keypoint adjustment
                h, w, _ = cropped_image.shape
                
                # Offset calculation should consider the relative position on the image
                y_offset = (max_height - h) // 2 - y_min  # y_min is the original image y_min of the bbox
                x_offset = (max_width - w) // 2 - x_min  # x_min is the original image x_min of the bbox

                adjusted_keypoint = adjust_keypoints(entry["coords"][class_name], x_offset, y_offset)


                # # Check if adjusted keypoints are out of the cropped images
                # if sum(map(lambda x: (x<((max_width-w)//2)) or (x>((max_width+w)//2)), adjusted_keypoint["x"])) >= 1:
                #     print(f"Some keypoints X out of cropped images: {cropped_filename}")
                # if sum(map(lambda y: (y<((max_height-h)//2)) or (y>((max_height+h)//2)), adjusted_keypoint["y"])) >= 1:
                #     print(f"Some keypoints Y out of cropped images: {cropped_filename}")


                # Check if adjusted keypoints are out of the bound
                if sum(map(lambda x: (x<0) or (x>max_width), adjusted_keypoint["x"])) >= 1:
                    print(f"Some keypoints X out of bound: {cropped_filename}")
                if sum(map(lambda y: (y<0) or (y>max_height), adjusted_keypoint["y"])) >= 1:
                    print(f"Some keypoints Y out of bound: {cropped_filename}")


                """ Uncomment if you want to save the visualization results of adjusted keypoints on cropped images """
                # # Draw adjusted keypoints on the black canvas image
                # draw_keypoints(canvas, adjusted_keypoint)
                # # Save the canvas-pasted image with keypoints
                # cropped_with_keypoints_output_folder = os.path.join(cropped_output_folder, "keypoints_visualization")
                # os.makedirs(cropped_with_keypoints_output_folder, exist_ok=True)
                # cropped_image_with_keypoints_path = os.path.join(cropped_with_keypoints_output_folder, cropped_filename)
                # cv2.imwrite(cropped_image_with_keypoints_path, canvas)


                adjusted_keypoint_list.append({
                    "filename"   : cropped_filename,
                    "class_name" : class_name,
                    "width"      : max_width,
                    "height"     : max_height,
                    "coords"     : adjusted_keypoint
                })

    # Save the adjusted keypoints
    keypoints_file_path = os.path.join("datasets", "MARS", f"MARS_cropped_keypoints_{view_type}_YOLO.json")
    with open(keypoints_file_path, "w") as kp_file:
        json.dump(adjusted_keypoint_list, kp_file, indent=4)
