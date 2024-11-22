import json
# import os

# # Set the working directory
# os.chdir('/Users/dahye/PycharmProjects/CLIP-pose-estimation')
#
# # Verify the current working directory
# print("Current working directory:", os.getcwd())

# Load the JSON data from the file
with open('datasets/MARS/MARS_keypoints_top.json') as f:
    input_data = json.load(f)
print(input_data[0])
print(input_data[0]['coords']['black'])
print(input_data[0]['coords']['white'])

# Function to compute average coordinates for a single set of keypoint
# Initialize accumulators for black and white datasets
total_black_coords_len, black_count = 0, 0
total_white_coords_len, white_count = 0, 0

# Iterate over the entire dataset
for entry in input_data:
    # For black coordinates
    if 'black' in entry['coords']:
        total_black_coords_len += len(entry['coords']['black']['x'])
        black_count += 1
    # For white coordinates
    if 'white' in entry['coords']:
        total_white_coords_len += len(entry['coords']['white']['x'])
        white_count += 1

# Calculate overall averages
avg_black_coords_cnt = total_black_coords_len / black_count
avg_white_coords_cnt = total_white_coords_len / white_count

# Print results
print(f"Average black keypoints count: {avg_black_coords_cnt}")
print(f"Average white keypoints count: {avg_white_coords_cnt}")
