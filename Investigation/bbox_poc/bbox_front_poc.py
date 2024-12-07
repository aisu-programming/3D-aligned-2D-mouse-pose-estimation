from PIL import Image, ImageDraw, ImageFont
"""
For top:
    nine anatomical keypoints in the top-view video 
    (the nose, ears, base of neck, hips, and tail base, midpoint, and endpoint)
For front:
    there could be 9 + (1-4) keypoints (top-view keypoints plus the four paws)
"""


image_path = "MARS_front_00000.jpg"

# Coordinates and labels
labels = [
    "nose tip",
    "right ear",
    "left ear",
    "neck",
    "right side body",
    "left side body",
    "tail base",
    "middle tail",
    "end tail"
]
coords_black = {
    "x": [
        1115.7, 1065.0, 1049.2, 1046.2, 1062.0,
        996.8, 986.7, 997.8, 1005.9, 1021.0, 1071.4
    ],
    "y": [
        8.1, 36.3, 33.0, 67.3, 138.1,
        201.8, 240.0, 261.1, 266.1, 198.8, 188.1
    ]
}

# Adjust labels for extra points
labels += ["extra_point"] * (len(coords_black["x"]) - len(labels))

try:
    # Open image and ensure RGB mode
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Load font
    font = None
    try:
        font_size = max(12, int(image.size[1] * 0.03))  # Dynamically adjust font size
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Draw keypoints and labels
    for i, (x, y) in enumerate(zip(coords_black["x"], coords_black["y"])):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red")
        draw.text((x + 5, y - 10), labels[i], fill="white", font=font)

    # Calculate bounding box with margin
    MARGIN = 30
    x_min, y_min = min(coords_black["x"]) - MARGIN, min(coords_black["y"]) - MARGIN
    x_max, y_max = max(coords_black["x"]) + MARGIN, max(coords_black["y"]) + MARGIN

    # Clip bounding box to image dimensions
    img_width, img_height = image.size
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)

    # Draw bounding box
    draw.rectangle([x_min, y_min, x_max, y_max], outline="yellow", width=3)

    # Show and save the annotated image
    image.show()
    output_path = "front_with_keypoints_and_bbox.jpg"
    image.save(output_path)
    print(f"Annotated image saved at: {output_path}")

except FileNotFoundError:
    print("Image file not found. Please provide a valid image path.")



