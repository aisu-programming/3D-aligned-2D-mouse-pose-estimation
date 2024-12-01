import os
import re

# Directory paths
top_dir = "datasets/MARS/cropped_images_top_YOLO/"
front_dir = "datasets/MARS/cropped_images_front_YOLO/"

# Extract <number>_<color> labels
def get_labels(directory, prefix):
    files = os.listdir(directory)
    labels = set()
    for f in files:
        match = re.search(rf"{prefix}_(\d+)_(black|white)\.jpg", f)
        if match:
            labels.add(f"{match.group(1)}_{match.group(2)}")
    return labels

# Get labels from the top and front directories
top_labels = get_labels(top_dir, "MARS_top")
front_labels = get_labels(front_dir, "MARS_front")

# Find unmatched labels
only_in_top = top_labels - front_labels
only_in_front = front_labels - top_labels

# Output the results
print("Only in top:", sorted(only_in_top))
print("Only in front:", sorted(only_in_front))
