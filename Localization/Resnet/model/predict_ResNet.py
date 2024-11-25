import os
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# ===========================
# Configuration Parameters
# ===========================

# Device configuration
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
LABELS_DIR = os.path.join(DATASET_DIR, 'labels')  # For ground truth
MODEL_DIR = os.path.join(BASE_DIR, '..', 'output')  # Changed to output directory
MODEL_PATH = os.path.join(MODEL_DIR, 'fasterrcnn_resnet50_fpn.pth')  # Correct path to the model
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output', 'predictions')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class Definitions
# Adjust these according to your training
# Original classes: 0 and 1
# After shifting: 1 and 2 (0 is background)
CLASS_ID_OFFSET = 1  # Shift class IDs by 1
CLASS_NAMES = {
    1: 'Class 0',  # Original class 0
    2: 'Class 1',  # Original class 1
}

# Confidence threshold for displaying predictions
CONFIDENCE_THRESHOLD = 0.5

# Number of images to predict
# Set to an integer value to limit the number of images
# Set to None to process all images
NUM_PREDICT_IMAGES = 20  # e.g., 100

# ===========================
# Utility Functions
# ===========================

def load_model(model_path, num_classes, device):
    """
    Load the trained Faster R-CNN model.
    """
    # Initialize the model without pretrained weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    
    # Replace the head with a new one for our classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_image_files(images_dir):
    """
    Retrieve sorted list of image file paths.
    """
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    return image_files, image_paths

def get_ground_truth(image_filename):
    """
    Load ground truth bounding boxes and labels from the corresponding label file.
    """
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    label_path = os.path.join(LABELS_DIR, label_filename)
    
    if not os.path.exists(label_path):
        print(f"Warning: Label file {label_filename} not found. Skipping ground truth for this image.")
        return []
    
    ground_truth = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Warning: Incorrect label format in {label_filename}. Expected 5 elements per line.")
                continue
            original_class_id = int(parts[0])
            class_id = original_class_id + CLASS_ID_OFFSET  # Shift class ID
            
            try:
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                print(f"Warning: Non-numeric values found in {label_filename}. Skipping this line.")
                continue
            
            # Convert normalized coordinates to absolute pixel values
            image_path = os.path.join(IMAGES_DIR, image_filename)
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            xmin = (x_center - width / 2) * img_width
            ymin = (y_center - height / 2) * img_height
            xmax = (x_center + width / 2) * img_width
            ymax = (y_center + height / 2) * img_height
            
            ground_truth.append({
                'bbox': [xmin, ymin, xmax, ymax],
                'label': class_id
            })
    
    return ground_truth

def visualize_predictions(image, ground_truth, detections, output_path):
    """
    Draw bounding boxes and labels on the image and save it.
    Displays both ground truth and predicted bounding boxes.
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Plot ground truth boxes
    for gt in ground_truth:
        bbox = gt['bbox']
        label = gt['label']
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        class_name = CLASS_NAMES.get(label, f"Class {label}")
        
        # Draw ground truth rectangle (green)
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='g', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # Add ground truth label
        ax.text(xmin, ymin - 10, f"GT: {class_name}", fontsize=12, color='g', weight='bold',
                verticalalignment='top')
    
    # Plot predicted boxes
    for det in detections:
        bbox = det['bbox']
        label = det['label']
        score = det['score']
        
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        class_name = CLASS_NAMES.get(label, f"Class {label}")
        
        # Draw predicted rectangle (red)
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add predicted label
        ax.text(xmin, ymin - 10, f"Pred: {class_name} ({score:.2f})", fontsize=12, color='r', weight='bold',
                verticalalignment='top')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='g', lw=2, linestyle='--', label='Ground Truth'),
        Line2D([0], [0], color='r', lw=2, label='Prediction')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def filter_detections(detections, confidence_threshold=0.5):
    """
    For each class, keep only the detection with the highest score.
    """
    filtered = {}
    for det in detections:
        label = det['label']
        score = det['score']
        bbox = det['bbox']
        
        if score < confidence_threshold:
            continue  # Skip detections below the threshold
        
        if label not in filtered or score > filtered[label]['score']:
            filtered[label] = {'bbox': bbox, 'score': score}
    
    # Convert to list of dictionaries
    filtered_list = [{'label': label, 'bbox': info['bbox'], 'score': info['score']} 
                     for label, info in filtered.items()]
    return filtered_list

# ===========================
# Main Prediction Function
# ===========================

def main():
    # Define number of classes (including background)
    num_classes = len(CLASS_NAMES) + 1  # +1 for background

    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
        return

    # Load the model
    print("Loading the model...")
    model = load_model(MODEL_PATH, num_classes, DEVICE)
    print("Model loaded successfully.")

    # Get list of images
    image_files, image_paths = get_image_files(IMAGES_DIR)
    total_images = len(image_paths)
    print(f"Found {total_images} images in {IMAGES_DIR}.")

    # Optionally limit the number of images to predict
    if NUM_PREDICT_IMAGES is not None:
        image_files = image_files[:NUM_PREDICT_IMAGES]
        image_paths = image_paths[:NUM_PREDICT_IMAGES]
        total_images = len(image_paths)
        print(f"Limiting predictions to the first {total_images} images.")

    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Iterate through images with progress bar
    for img_file, img_path in tqdm(zip(image_files, image_paths), total=total_images, desc="Predicting"):
        # Load image
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).to(DEVICE)
        
        # Perform inference
        with torch.no_grad():
            outputs = model([image_tensor])[0]
        
        # Extract detections
        detections = []
        for bbox, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
            detections.append({
                'bbox': bbox.cpu().numpy(),
                'label': label.item(),
                'score': score.item()
            })
        
        # Filter detections: keep only the highest scoring detection per class
        filtered_detections = filter_detections(detections, CONFIDENCE_THRESHOLD)
        
        # Load ground truth
        ground_truth = get_ground_truth(img_file)
        
        if not filtered_detections and not ground_truth:
            continue  # No detections and no ground truth
        
        # Draw and save the predictions along with ground truth
        output_image_path = os.path.join(OUTPUT_DIR, f"pred_{img_file}")
        visualize_predictions(image, ground_truth, filtered_detections, output_image_path)

    print(f"Prediction and visualization completed. Results saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
