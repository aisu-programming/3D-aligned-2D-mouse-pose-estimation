from PIL import Image, ImageDraw, ImageFont
"""
For top:
    nine anatomical keypoints in the top-view video 
    (the nose, ears, base of neck, hips, and tail base, midpoint, and endpoint)
For front:
    there could be 9 + (1-4) keypoints (top-view keypoints plus the four paws)
"""


image_path = "MARS_top_00000.jpg"

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
            1012.0,
            934.0,
            965.0,
            931.0,
            859.0,
            890.0,
            812.0
        ],
        "y": [
            382.0,
            372.0,
            308.0,
            322.0,
            349.0,
            258.0,
            294.0
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
    MARGIN = 25
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
    output_path = "top_with_keypoints_and_bbox.jpg"
    image.save(output_path)
    print(f"Annotated image saved at: {output_path}")

except FileNotFoundError:
    print("Image file not found. Please provide a valid image path.")



