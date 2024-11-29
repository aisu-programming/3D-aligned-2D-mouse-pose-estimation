import json
import os
# Set the working directory
os.chdir('/Users/dahye/PycharmProjects/CLIP-pose-estimation')

# Verify the current working directory
print("Current working directory:", os.getcwd())

#Load the JSON data from the file
with open('datasets/MARS/MARS_keypoints_front.json') as f:
    dataset = json.load(f)

# Initialize flags
all_match = True
for entry in dataset:
    if entry.get("width") != 1280 or entry.get("height") != 500:
        print(f"Mismatch in file: {entry['filename']}")
        print(f"Width: {entry['width']}, Height: {entry['height']}")
        all_match = False

# Final result
if all_match:
    print("All entries have width=1280 and height=500.")
else:
    print("Some entries do not match the required dimensions.")


#Load the JSON data from the file
with open('datasets/MARS/MARS_keypoints_top.json') as f:
    dataset = json.load(f)

# Initialize flags
all_match = True
for entry in dataset:
    if entry.get("width") != 1024 or entry.get("height") != 570:
        print(f"Mismatch in file: {entry['filename']}")
        print(f"Width: {entry['width']}, Height: {entry['height']}")
        all_match = False

# Final result
if all_match:
    print("TOP: All entries have width=1024 and height=570.")
else:
    print("TOP: Some entries do not match the required dimensions.")
