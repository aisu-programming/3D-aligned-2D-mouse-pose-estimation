import os
import torch
torch.manual_seed(42)
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from train.utils import get_lr, generate_heatmaps, visualize_peaks_from_heatmap, visualize_heatmaps, visualize_peaks
from models.hrnet import PoseHighResolutionNet
from data.dataset import Resize, Pad, ToTensor, RandomHorizontalFlip, RandomRotate90Degree, KeypointsDataset, CroppedKeypointsDataset
from torch.utils.data import random_split, DataLoader



FRONT = "front"
TOP = "top"

front_or_top = FRONT
sigma = 3
batch_size = 32
gamma = 0.99935
learning_rate = 0.0006
num_epochs = 200
num_block = 3
num_chnl_base = 24
chnl_exp_factor = 2.5



transform = transforms.Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomRotate90Degree(),
    ToTensor(),
])
dataset = CroppedKeypointsDataset(
    annotations_file=f"datasets/MARS/MARS_cropped_keypoints_{front_or_top}_YOLO.json",
    img_dir=f"datasets/MARS/cropped_images_{front_or_top}_YOLO",
    transform=transform)

dataset_size = len(dataset)
train_ratio = 0.8
train_size = int(train_ratio * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

cfg = {
    "NUM_JOINTS": 11 if front_or_top == FRONT else 7,
    "EXTRA": {
        "FINAL_CONV_KERNEL": 1,
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [ num_block for _ in range(2) ],
            "NUM_CHANNELS": [ int(num_chnl_base*(chnl_exp_factor**i)) for i in range(2) ],
            "FUSE_METHOD": "SUM"
        },
        "STAGE3": {
            "NUM_MODULES": 4,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [ num_block for _ in range(3) ],
            "NUM_CHANNELS": [ int(num_chnl_base*(chnl_exp_factor**i)) for i in range(3) ],
            "FUSE_METHOD": "SUM"
        },
        "STAGE4": {
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [ num_block for _ in range(4) ],
            "NUM_CHANNELS": [ int(num_chnl_base*(chnl_exp_factor**i)) for i in range(4) ],
            "FUSE_METHOD": "SUM"
        }
    }
}

model = PoseHighResolutionNet(cfg).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

os.makedirs("results", exist_ok=True)
os.makedirs("results/hrnet", exist_ok=True)
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for batch in tqdm(train_loader, desc=f"Epoch: {epoch+1}/{num_epochs} [TRAIN]"):
        images = batch["image"].to(device)
        keypoints = batch["keypoints"].to(device)
        outputs = model(images)

        heatmap_size = outputs.shape[2:]
        scale = outputs.shape[-1] / images.shape[-1]
        keypoints_heatmaps = generate_heatmaps(keypoints, heatmap_size=heatmap_size, sigma=sigma, scale=scale)
        loss: torch.Tensor = criterion(outputs, keypoints_heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_losses.append(loss.item())
    avg_train_loss = np.average(train_losses)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader),
                                desc=f"Epoch: {epoch+1}/{num_epochs} [VALID]",
                                total=len(val_loader)):
            images = batch["image"].to(device)
            keypoints = batch["keypoints"].to(device)
            outputs = model(images)

            heatmap_size = outputs.shape[2:]
            scale = outputs.shape[-1] / images.shape[-1]
            keypoints_heatmaps = generate_heatmaps(keypoints, heatmap_size=heatmap_size, sigma=sigma, scale=scale)
            loss = criterion(outputs, keypoints_heatmaps)

            val_losses.append(loss.item())
            scale = 1 / scale
            if idx == 0:
                for i in range(images.size(0)):
                    if i == 8: break
                    image = images[i]
                    visualize_heatmaps(image, outputs[i, :1], save_path=f"results/hrnet/heatmap_pred_{idx}_{i}.png")
                    visualize_heatmaps(image, keypoints_heatmaps[i, :1], save_path=f"results/hrnet/heatmap_truth_{idx}_{i}.png")
                    visualize_peaks_from_heatmap(image, outputs[i], save_path=f"results/hrnet/peaks_pred_{idx}_{i}.png", scale=scale)
                    visualize_peaks_from_heatmap(image, keypoints_heatmaps[i], save_path=f"results/hrnet/peaks_truth_{idx}_{i}.png", scale=scale)

    avg_val_loss = np.average(val_losses)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.8f}, " + \
            f"Validation Loss: {avg_val_loss:.8f}, LR: {get_lr(optimizer):.10f}")
    
    with open("results/hrnet/logs.txt", mode='a') as log_file:
        log_file.write(f"{epoch},{avg_train_loss},{avg_val_loss},{get_lr(optimizer)}\n")
    