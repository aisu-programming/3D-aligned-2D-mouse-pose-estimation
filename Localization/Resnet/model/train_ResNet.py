import os
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# ===========================
# Configuration Parameters
# ===========================
# Number of images to use for the entire dataset (set to None to use all)
NUM_IMAGES = None  # e.g., 1000

# Split ratio for train and test sets
TRAIN_SPLIT = 0.8  # 80% training, 20% testing

# Batch size for training and testing
BATCH_SIZE = 16

# Number of epochs
NUM_EPOCHS = 5

# Learning rate
LEARNING_RATE = 0.005

# Device configuration
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Paths
DATASET_DIR = os.path.join('..', 'dataset')
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
LABELS_DIR = os.path.join(DATASET_DIR, 'labels')
OUTPUT_DIR = os.path.join('..', 'output')
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
OUTPUT_IMAGES_DIR = os.path.join(PREDICTIONS_DIR, 'images')
OUTPUT_LABELS_DIR = os.path.join(PREDICTIONS_DIR, 'labels')
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, 'loss_log.txt')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# ===========================
# Class Definitions
# ===========================

# Define a mapping from original class IDs to model class IDs
# Original classes: 0 and 1
# Model expects classes: 1 and 2
CLASS_ID_OFFSET = 1  # Shift class IDs by 1

# Define class names for visualization
CLASS_NAMES = {
    1: 'Class 0',  # Original class 0
    2: 'Class 1',  # Original class 1
}

# ===========================
# Custom Dataset Class
# ===========================
class LocalizationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, image_files=None):
        """
        Args:
            images_dir (str): Directory with all the images.
            labels_dir (str): Directory with all the labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            image_files (list, optional): List of image filenames to include.
                If None, all images in the directory are used.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        if image_files is not None:
            self.image_files = sorted(image_files)
        else:
            self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Load labels
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpg', '.txt'))
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                original_class_id = int(parts[0])
                class_id = original_class_id + CLASS_ID_OFFSET  # Shift class ID

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert to absolute coordinates
                xmin = (x_center - width / 2) * img_width
                ymin = (y_center - height / 2) * img_height
                xmax = (x_center + width / 2) * img_width
                ymax = (y_center + height / 2) * img_height

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)

        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)

        return img, target

# ===========================
# Utility Functions
# ===========================
def collate_fn(batch):
    return tuple(zip(*batch))

def visualize_predictions(image, ground_truth, predictions, output_path):
    """
    Visualize ground truth and predicted bounding boxes on the image and save it.
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    # Convert tensor to PIL Image
    image = image.permute(1, 2, 0).cpu().numpy()
    ax.imshow(image)

    # Plot ground truth boxes
    for box, label in zip(ground_truth['boxes'], ground_truth['labels']):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        class_name = CLASS_NAMES.get(label.item(), f"Class {label.item()}")
        rect = patches.Rectangle((xmin, ymin), width, height,
                                 linewidth=2, edgecolor='g', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f"GT: {class_name}", fontsize=12, color='g', verticalalignment='top')

    # Plot predicted boxes
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= 0.5:  # Threshold for visualization
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            class_name = CLASS_NAMES.get(label.item(), f"Class {label.item()}")
            rect = patches.Rectangle((xmin, ymin), width, height,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 10, f"Pred: {class_name} ({score:.2f})", fontsize=12, color='r',
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

def save_predictions(detections, image_size, output_txt_path):
    """
    Save detections to a .txt file in the format:
    class_id x_center y_center width height
    where coordinates are normalized between 0 and 1.
    """
    img_width, img_height = image_size
    lines = []
    for det in detections:
        label = det['label']
        bbox = det['bbox']
        score = det['score']  # Optional: You can choose to include the score or not

        xmin, ymin, xmax, ymax = bbox
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Ensure values are between 0 and 1
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width = min(max(width, 0.0), 1.0)
        height = min(max(height, 0.0), 1.0)

        line = f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        lines.append(line)

    # Write to .txt file
    with open(output_txt_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    # If no detections, an empty file is created

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected to be in [xmin, ymin, xmax, ymax] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) != 0 else 0

    return iou

# ===========================
# Evaluation Function
# ===========================
def evaluate(model, data_loader, device):
    """
    Evaluate the model on the test dataset and compute precision and MSE for bounding boxes.
    Ensures only one prediction per object is used (highest confidence).
    Returns precision and MSE.
    """
    model.eval()
    total = 0
    correct = 0
    total_mse = 0.0
    total_matched = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating", leave=False):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes = target['boxes'].cpu().numpy()
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()

                matched_gt = set()
                matched_pred = set()

                # Sort predictions by score descending
                sorted_indices = np.argsort(-pred_scores)
                pred_boxes = pred_boxes[sorted_indices]
                pred_scores = pred_scores[sorted_indices]
                pred_labels = pred_labels[sorted_indices]

                for pred_idx, (pred_box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                    if score < 0.5:
                        continue
                    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, target['labels'].cpu().numpy())):
                        if gt_label != label or gt_idx in matched_gt:
                            continue
                        iou = compute_iou(pred_box, gt_box)
                        if iou >= 0.5:
                            correct += 1
                            mse = np.mean((pred_box - gt_box) ** 2)
                            total_mse += mse
                            total_matched += 1
                            matched_gt.add(gt_idx)
                            matched_pred.add(pred_idx)
                            break  # Move to the next prediction

                total += len(gt_boxes)

    precision = correct / total if total > 0 else 0
    mse = total_mse / total_matched if total_matched > 0 else 0.0

    return precision, mse

# ===========================
# Main Training Function
# ===========================
def main():
    # Initialize the log file
    with open(LOG_FILE_PATH, 'w') as log_file:
        log_file.write("Epoch,Average_Training_Loss,Validation_Precision,Validation_MSE\n")

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Initialize the full dataset
    full_image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')])

    if NUM_IMAGES is not None:
        full_image_files = full_image_files[:NUM_IMAGES]

    # Create the full dataset
    full_dataset = LocalizationDataset(IMAGES_DIR, LABELS_DIR, transform=transform, image_files=full_image_files)

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(TRAIN_SPLIT * total_size)
    test_size = total_size - train_size

    # Ensure that the dataset is large enough
    if train_size == 0 or test_size == 0:
        raise ValueError("Train/Test split results in an empty dataset. Adjust NUM_IMAGES or TRAIN_SPLIT.")

    # Split the dataset into training and testing
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(42))  # For reproducibility

    print(f"Total images: {total_size}")
    print(f"Training images: {train_size}")
    print(f"Testing images: {test_size}")

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, collate_fn=collate_fn)

    # Initialize the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Determine the number of classes
    # Original class IDs: 0 and 1
    # After shifting: 1 and 2
    num_classes = 3  # 2 classes + background

    # Replace the classifier with a new one, that has num_classes which is user-defined
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(DEVICE)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE,
                                momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', leave=False)
        for images, targets in loop:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            loop.set_postfix(loss=losses.item())

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Average Loss: {avg_loss:.4f}")

        # Evaluate on the test set
        precision, mse = evaluate(model, test_loader, DEVICE)
        print(f"Validation Precision: {precision:.4f}, Validation MSE: {mse:.4f}")

        # Log the metrics
        with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(f"{epoch+1},{avg_loss:.4f},{precision:.4f},{mse:.4f}\n")

    # Save the trained model
    model_path = os.path.join(OUTPUT_DIR, 'fasterrcnn_resnet50_fpn.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Final Prediction and Visualization on Test Set
    print("Performing final predictions on the test set...")

    # Create a DataLoader for the entire test set with batch_size=1 for visualization
    visualize_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(visualize_loader, desc="Visualizing Predictions")):
            image = images[0].to(DEVICE)
            target = targets[0]
            prediction = model([image])[0]

            # Move tensors to CPU for visualization
            image_cpu = image.cpu()
            target_cpu = {k: v.cpu() for k, v in target.items()}
            prediction_cpu = {k: v.cpu() for k, v in prediction.items()}

            # Match predictions to ground truths to ensure only one prediction per object
            matched_predictions = match_predictions_to_ground_truths(target_cpu, prediction_cpu)

            # Save the image with bounding boxes
            image_filename = test_dataset.dataset.image_files[test_dataset.indices[idx]]
            output_image_path = os.path.join(OUTPUT_IMAGES_DIR, f"pred_{image_filename}")
            visualize_predictions(image_cpu, target_cpu, matched_predictions, output_image_path)

            # Save predictions to .txt file
            output_txt_filename = f"pred_{os.path.splitext(image_filename)[0]}.txt"
            output_txt_path = os.path.join(OUTPUT_LABELS_DIR, output_txt_filename)
            img_width, img_height = Image.open(os.path.join(IMAGES_DIR, image_filename)).size
            detections = []
            for box, label, score in zip(matched_predictions['boxes'], matched_predictions['labels'], matched_predictions['scores']):
                detections.append({
                    'bbox': box,
                    'label': label.item(),
                    'score': score.item()
                })
            save_predictions(detections, (img_width, img_height), output_txt_path)

    print(f"Prediction and visualization completed. Results saved to {PREDICTIONS_DIR}")

# ===========================
# Matching Function
# ===========================
def match_predictions_to_ground_truths(ground_truth, predictions, iou_threshold=0.5):
    """
    Matches predictions to ground truths ensuring only one prediction per ground truth.
    Selects the prediction with the highest confidence score for each ground truth.
    Returns the filtered predictions.
    """
    gt_boxes = ground_truth['boxes'].numpy()
    gt_labels = ground_truth['labels'].numpy()

    pred_boxes = predictions['boxes'].numpy()
    pred_labels = predictions['labels'].numpy()
    pred_scores = predictions['scores'].numpy()

    matched_gt = set()
    matched_pred = set()
    filtered_predictions = {'boxes': [], 'labels': [], 'scores': []}

    # Sort predictions by score descending
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    for pred_idx, (pred_box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        if score < 0.5:
            continue  # Skip low confidence predictions
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_label != label or gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                # Assign this prediction to the ground truth
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                filtered_predictions['boxes'].append(pred_box)
                filtered_predictions['labels'].append(label)
                filtered_predictions['scores'].append(score)
                break  # Move to the next prediction

    return filtered_predictions

if __name__ == '__main__':
    main()
