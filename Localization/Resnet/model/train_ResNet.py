import os
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# ===========================
# Configuration Parameters
# ===========================
# Number of images to use for training (set to None to use all)
NUM_TRAIN_IMAGES = 150  # e.g., 1000

# Batch size for training
BATCH_SIZE = 4

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
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# Class Definitions
# ===========================

# Define a mapping from original class IDs to model class IDs
# Original classes: 0 and 1
# Model expects classes: 1 and 2
CLASS_ID_OFFSET = 1  # Shift class IDs by 1

# Optional: Define class names for visualization
CLASS_NAMES = {
    1: 'Class 0',  # Original class 0
    2: 'Class 1',  # Original class 1
}

# ===========================
# Custom Dataset Class
# ===========================
class LocalizationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, num_images=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

        if num_images is not None:
            self.image_files = self.image_files[:num_images]

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
    image = image.permute(1, 2, 0).numpy()
    ax.imshow(image)

    # Plot ground truth boxes
    for box, label in zip(ground_truth['boxes'], ground_truth['labels']):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        class_name = CLASS_NAMES.get(label.item(), f"Class {label.item()}")
        rect = patches.Rectangle((xmin, ymin), width, height,
                                 linewidth=2, edgecolor='g', facecolor='none')
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

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# ===========================
# Main Training Function
# ===========================
def main():
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    dataset = LocalizationDataset(IMAGES_DIR, LABELS_DIR, transform=transform, num_images=NUM_TRAIN_IMAGES)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

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
        loop = tqdm(data_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', leave=False)
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
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Average Loss: {avg_loss:.4f}")

    # Save the trained model
    model_path = os.path.join(OUTPUT_DIR, 'fasterrcnn_resnet50_fpn.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluation and Visualization
    model.eval()
    visualize_count = 5  # Number of images to visualize
    visualize_dataset = LocalizationDataset(IMAGES_DIR, LABELS_DIR, transform=transform, num_images=visualize_count)
    visualize_loader = DataLoader(visualize_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(visualize_loader, desc="Visualizing Predictions")):
            if idx >= visualize_count:
                break
            image = images[0].to(DEVICE)
            target = targets[0]
            prediction = model([image])[0]

            # Move tensors to CPU for visualization
            image_cpu = image.cpu()
            target_cpu = {k: v.cpu() for k, v in target.items()}
            prediction_cpu = {k: v.cpu() for k, v in prediction.items()}

            # Save the image with bounding boxes
            image_filename = visualize_dataset.image_files[idx]
            output_path = os.path.join(OUTPUT_DIR, f"pred_{image_filename}")
            visualize_predictions(image_cpu, target_cpu, prediction_cpu, output_path)

    print(f"Visualization images saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
