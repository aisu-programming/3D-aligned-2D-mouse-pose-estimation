import json
import os

# Load your custom JSON file
with open('MARS_front_raw.json', 'r') as f:
    custom_data = json.load(f)

# Define the relative path to the images directory
relative_image_path = "raw_images_front/"  # Update this to the relative folder where your images are stored

coco_data = {
    "info": {
        "description": "Your dataset",
        "version": "1.0",
        "year": 2024
    },
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "animal",
            "supercategory": "pose",
            "keypoints": [
                "nose tip",
                "right ear",
                "left ear",
                "neck",
                "right side body",
                "left side body",
                "tail base",
                "middle tail",
                "end tail"
            ],
            "skeleton": []
        }
    ]
}

# Loop through each image in your custom JSON
for img_id, img_data in enumerate(custom_data):
    # Add image metadata to COCO, using relative paths
    coco_data["images"].append({
        "file_name": os.path.join(relative_image_path, img_data["filename"]),  # Relative path
        "height": img_data["height"],
        "width": img_data["width"],
        "id": img_id + 1
    })

    # Loop through each subject (e.g., "black" and "white")
    for subject, coords in img_data["coords"].items():
        keypoints = []
        num_keypoints = 0
        labels = img_data["labels"]
        
        # Build the keypoints array for COCO format
        for i in range(len(labels)):
            x = coords["x"][i]
            y = coords["y"][i]
            v = 2  # Visibility, you can modify this based on your data

            keypoints.extend([x, y, v])
            if v > 0:
                num_keypoints += 1
        
        # Add the annotation
        coco_data["annotations"].append({
            "id": len(coco_data["annotations"]) + 1,
            "image_id": img_id + 1,
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": num_keypoints,
            "bbox": [],  # Optional: You can calculate bounding boxes if needed
            "segmentation": [],  # Optional: Leave this blank if not needed
            "area": 0  # Optional: Leave this as 0 if not needed
        })

# Save the COCO formatted data to a new JSON file
with open('MARS_front_COCO.json', 'w') as f:
    json.dump(coco_data, f, indent=4)

print("Conversion complete! COCO format saved as Mars_coco_annotations.json")
