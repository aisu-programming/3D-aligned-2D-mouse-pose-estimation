import os
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from utils import get_lr, generate_heatmaps, visualize_heatmaps, visualize_peaks
from models.unet import UNet
from data.dataset import Resize, Pad, ToTensor, RandomHorizontalFlip, CocoKeypointsDataset
from torch.utils.data import random_split, DataLoader



base_filters = 32
batch_size = 16
dropout_prob = 0.1
expand_factor = 2
gamma = 0.9994
learning_rate = 0.0016
num_groups = 4
num_layers = 5
sigma = 10



transform = transforms.Compose([
    Resize((125, 320)),
    Pad((128, 320)),
    RandomHorizontalFlip(p=0.5, img_width=320),
    ToTensor(),
])

dataset = CocoKeypointsDataset(
    img_dir="datasets/MARS",
    ann_file="datasets/MARS/MARS_front_COCO.json",
    transform=transform)

dataset_size = len(dataset)
train_ratio = 0.8
train_size = int(train_ratio * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
model = UNet(n_channels=1, n_classes=18,
             base_filters=base_filters,
             num_layers=num_layers,
             expand_factor=expand_factor,
             num_groups=num_groups,
             dropout_prob=dropout_prob).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

os.makedirs("results", exist_ok=True)
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for batch in tqdm(train_loader, desc=f"Epoch: {epoch+1}/{num_epochs} [TRAIN]"):
        images = batch["image"].to(device)
        keypoints = batch["keypoints"].to(device)
        outputs = model(images)
        keypoints_heatmaps = generate_heatmaps(keypoints, heatmap_size=outputs.shape[2:], sigma=sigma)
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
        for idx, batch in tqdm(enumerate(val_loader), desc=f"Epoch: {epoch+1}/{num_epochs} [VALID]", total=len(val_loader)):
            images = batch["image"].to(device)
            keypoints = batch["keypoints"].to(device)
            outputs = model(images)
            keypoints_heatmaps = generate_heatmaps(keypoints, heatmap_size=outputs.shape[2:], sigma=sigma)
            loss = criterion(outputs, keypoints_heatmaps)
            val_losses.append(loss.item())
            if idx == 0:
                for i in range(images.size(0)):
                    if i == 8: break
                    image = images[i]

                    visualize_heatmaps(image, outputs[i], save_path=f"results/heatmap_pred_{idx}_{i}.png")
                    visualize_peaks(image, outputs[i], save_path=f"results/peaks_pred_{idx}_{i}.png")

                    visualize_heatmaps(image, keypoints_heatmaps[i], save_path=f"results/heatmap_truth_{idx}_{i}.png")
                    visualize_peaks(image, keypoints_heatmaps[i], save_path=f"results/peaks_truth_{idx}_{i}.png")

    avg_val_loss = np.average(val_losses)

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.8f}, Validation Loss: {avg_val_loss:.8f}, LR: {get_lr(optimizer):.10f}")


# with torch.no_grad():
#     for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
#         images = batch["image"].to(device)
#         keypoints = batch["keypoints"].to(device)
#         # outputs = model(images)
#         outputs = generate_heatmaps(keypoints, heatmap_size=images.shape[2:], sigma=5)
#         for i in range(images.size(0)):
#             image = images[i]
#             heatmaps = outputs[i]
#             save_path_heatmap = f"results/heatmap_overlay_{idx}_{i}.png"
#             visualize_heatmaps(image, heatmaps, save_path=save_path_heatmap)
#             save_path_peaks = f"results/peaks_overlay_{idx}_{i}.png"
#             visualize_peaks(image, heatmaps, save_path=save_path_peaks)
#         break