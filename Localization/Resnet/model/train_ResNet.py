import os
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# ===========================
# Configuration Parameters
# ===========================

# Number of images to use for training and validation (set to None to use all)
NUM_TRAIN_IMAGES = None  # e.g., 800
NUM_VAL_IMAGES = None    # e.g., 200

# Batch size for training and testing
BATCH_SIZE = 16

# Number of epochs
NUM_EPOCHS = 1

# Learning rate
LEARNING_RATE = 0.005

# Device configuration
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Paths
DATASET_DIR = os.path.join('..', 'dataset')
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, 'images', 'train')
TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, 'labels', 'train')
VAL_IMAGES_DIR = os.path.join(DATASET_DIR, 'images', 'val')
VAL_LABELS_DIR = os.path.join(DATASET_DIR, 'labels', 'val')
OUTPUT_DIR = os.path.join('..', 'output')
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, 'loss_log_test.txt')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# Class Definitions
# ===========================

# Define a mapping from original class IDs to model class IDs
# Original classes: 0 and 1
# Model expects classes: 1 and 2
CLASS_ID_OFFSET = 1  # Shift class IDs by 1

# Define class names for visualization (optional, can be removed if not visualizing)
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
        if os.path.exists(label_path):
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
    Evaluate the model on the validation dataset and compute precision and MSE for bounding boxes.
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
                        continue  # Skip low confidence predictions
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

    # Initialize the training dataset
    train_image_files = sorted([f for f in os.listdir(TRAIN_IMAGES_DIR) if f.endswith('.jpg')])

    if NUM_TRAIN_IMAGES is not None:
        if NUM_TRAIN_IMAGES > len(train_image_files):
            print(f"Requested {NUM_TRAIN_IMAGES} training images, but only {len(train_image_files)} available. Using all training images.")
            train_image_files = train_image_files
        else:
            train_image_files = train_image_files[:NUM_TRAIN_IMAGES]

    train_dataset = LocalizationDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, transform=transform, image_files=train_image_files)

    # Initialize the validation dataset
    val_image_files = sorted([f for f in os.listdir(VAL_IMAGES_DIR) if f.endswith('.jpg')])

    if NUM_VAL_IMAGES is not None:
        if NUM_VAL_IMAGES > len(val_image_files):
            print(f"Requested {NUM_VAL_IMAGES} validation images, but only {len(val_image_files)} available. Using all validation images.")
            val_image_files = val_image_files
        else:
            val_image_files = val_image_files[:NUM_VAL_IMAGES]

    val_dataset = LocalizationDataset(VAL_IMAGES_DIR, VAL_LABELS_DIR, transform=transform, image_files=val_image_files)

    print(f"Total training images: {len(train_dataset)}")
    print(f"Total validation images: {len(val_dataset)}")

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
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

        # Evaluate on the validation set
        precision, mse = evaluate(model, val_loader, DEVICE)
        print(f"Validation Precision: {precision:.4f}, Validation MSE: {mse:.4f}")

        # Log the metrics
        with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(f"{epoch+1},{avg_loss:.4f},{precision:.4f},{mse:.4f}\n")

    # Save the trained model
    model_path = os.path.join(OUTPUT_DIR, 'fasterrcnn_resnet50_fpn_test.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
