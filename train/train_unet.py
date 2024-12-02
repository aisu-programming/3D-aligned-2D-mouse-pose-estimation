import os
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from train.utils import get_lr, generate_heatmaps, visualize_heatmaps, visualize_peaks_from_heatmap, visualize_peaks
from models.unet import UNet
from data.dataset import Resize, Pad, ToTensor, RandomHorizontalFlip, RandomRotate90Degree, KeypointsDataset, CroppedKeypointsDataset
from torch.utils.data import random_split, DataLoader



HEATMAP_OR_PEAK = "heatmap"
TOP_OR_FRONT = "front"
SIGMA = 30

base_filters = 48
batch_size = 16
dropout_prob = 0.2
expand_factor = 3
gamma = 0.9999
learning_rate = 0.005
num_groups = 4
num_layers = 4
sigma = 30
num_epochs = 100



# transform = transforms.Compose([
#     Resize((125, 320)),
#     Pad((128, 320)),
#     RandomHorizontalFlip(),
#     RandomRotate90Degree(),
#     ToTensor(),
# ])
# dataset = KeypointsDataset(
#     annotations_file="datasets/MARS/MARS_keypoints_front.json",
#     img_dir="datasets/MARS/raw_images_front",
#     transform=transform)

if TOP_OR_FRONT == "top":
    transform = transforms.Compose([
        Resize((240, 240)),
        # Pad((128, 320)),
        RandomHorizontalFlip(),
        RandomRotate90Degree(),
        ToTensor(),
    ])
elif TOP_OR_FRONT == "front":
    transform = transforms.Compose([
        Resize((240, 240)),
        # Pad((128, 320)),
        RandomHorizontalFlip(),
        RandomRotate90Degree(),
        ToTensor(),
    ])
else:
    raise Exception

dataset = CroppedKeypointsDataset(
    annotations_file=f"datasets/MARS/MARS_cropped_keypoints_{TOP_OR_FRONT}_YOLO.json",
    img_dir=f"datasets/MARS/cropped_images_{TOP_OR_FRONT}_YOLO",
    transform=transform)

dataset_size = len(dataset)
train_ratio = 0.8
train_size = int(train_ratio * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)

model = UNet(n_channels=1,
             n_classes=9 if TOP_OR_FRONT == "top" else 11,
             base_filters=base_filters,
             num_layers=num_layers,
             expand_factor=expand_factor,
             num_groups=num_groups,
             dropout_prob=dropout_prob).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

os.makedirs("results", exist_ok=True)
for epoch in range(num_epochs):

    model.train()
    train_losses = []
    for batch in tqdm(train_loader, desc=f"Epoch: {epoch+1}/{num_epochs} [TRAIN]"):
        images: torch.Tensor = batch["image"].to(device)
        keypoints: torch.Tensor = batch["keypoints"].to(device)
        outputs = model(images)

        if HEATMAP_OR_PEAK == "heatmap":
            keypoints_heatmaps = generate_heatmaps(keypoints, heatmap_size=outputs.shape[2:], sigma=SIGMA)
            loss: torch.Tensor = criterion(outputs, keypoints_heatmaps)
        else:
            batch_size, num_joints, _, _ = outputs.shape
            keypoints_pred = []
            for b in range(batch_size):
                keypoints_pred_batch = []
                for i in range(num_joints):
                    heatmap = outputs[b, i].cpu().detach().numpy()
                    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    keypoints_pred_batch.append([x, y, 2])
                keypoints_pred.append(keypoints_pred_batch)
            keypoints_pred = torch.tensor(keypoints_pred, dtype=torch.float32, requires_grad=True).to(device)
            loss: torch.Tensor = criterion(keypoints_pred, keypoints)

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

            if HEATMAP_OR_PEAK == "heatmap":
                keypoints_heatmaps = generate_heatmaps(keypoints, heatmap_size=outputs.shape[2:], sigma=SIGMA)
                loss = criterion(outputs, keypoints_heatmaps)
            else:
                batch_size, num_joints, _, _ = outputs.shape
                keypoints_pred = []
                for b in range(batch_size):
                    keypoints_pred_batch = []
                    for i in range(num_joints):
                        heatmap = outputs[b, i].cpu().detach().numpy()
                        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                        keypoints_pred_batch.append([x, y, 2])
                    keypoints_pred.append(keypoints_pred_batch)
                keypoints_pred = torch.tensor(keypoints_pred, dtype=torch.float32, requires_grad=True).to(device)
                loss: torch.Tensor = criterion(keypoints_pred, keypoints)

            val_losses.append(loss.item())
            if idx == 0:
                for i in range(images.size(0)):
                    if i == 8: break
                    image = images[i]

                    if HEATMAP_OR_PEAK == "heatmap":
                        # visualize_heatmaps(image, outputs[i], save_path=f"results/heatmap_pred_{idx}_{i}.png")
                        # visualize_heatmaps(image, keypoints_heatmaps[i], save_path=f"results/heatmap_truth_{idx}_{i}.png")
                        visualize_peaks_from_heatmap(image, outputs[i], save_path=f"results/peaks_pred_{idx}_{i}.png")
                        visualize_peaks_from_heatmap(image, keypoints_heatmaps[i], save_path=f"results/peaks_truth_{idx}_{i}.png")
                    else:
                        visualize_peaks(image, keypoints_pred[i], save_path=f"results/peaks_pred_{idx}_{i}.png")
                        visualize_peaks(image, keypoints[i], save_path=f"results/peaks_truth_{idx}_{i}.png")

    avg_val_loss = np.average(val_losses)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.8f}, Validation Loss: {avg_val_loss:.8f}, LR: {get_lr(optimizer):.10f}")


# with torch.no_grad():
#     for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
#         images = batch["image"].to(device)
#         keypoints = batch["keypoints"].to(device)
#         # outputs = model(images)
#         outputs = generate_heatmaps(keypoints, heatmap_size=images.shape[2:], sigma=SIGMA)
#         for i in range(images.size(0)):
#             image = images[i]
#             heatmaps = outputs[i]
#             save_path_heatmap = f"results/heatmap_overlay_{idx}_{i}.png"
#             visualize_heatmaps(image, heatmaps, save_path=save_path_heatmap)
#             save_path_peaks = f"results/peaks_overlay_{idx}_{i}.png"
#             visualize_peaks(image, heatmaps, save_path=save_path_peaks)
#         break
