import os
import torch
torch.manual_seed(42)
import wandb
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from train.utils import get_lr, generate_heatmaps, visualize_heatmaps, visualize_peaks_from_heatmap
from models.unet import UNet
from data.dataset import Resize, Pad, ToTensor, RandomHorizontalFlip, RandomRotate90Degree, KeypointsDataset, CroppedKeypointsDataset
from torch.utils.data import Subset, DataLoader



FRONT = "front"
TOP = "top"

front_or_top = FRONT
sigma = 12
training_dataset_size = 300

wandb.init(
    project="TDPE_UNet_vs_CL_UNet_TDS",
    name=f"UNet_{front_or_top}_tds={training_dataset_size}",
    config={
        "base_filters": 16,
        "batch_size": 16,
        "dropout_prob": 0.01,
        "expand_factor": 2,
        "gamma": 0.99991,
        "learning_rate": 0.006,
        "num_groups": 8,
        "num_layers": 6,
        "num_epochs": 300,
        "training_dataset_size": training_dataset_size,
    }
)

config = wandb.config

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

validation_dataset_size = len(dataset) - training_dataset_size
print("The size of entire dataset:", len(dataset))
print("The size of training dataset:", training_dataset_size)
print("The size of validation dataset:", validation_dataset_size, "\n")
val_dataset = Subset(dataset, range(validation_dataset_size))
train_dataset = Subset(dataset, range(validation_dataset_size, len(dataset)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size*2, shuffle=False)

model = UNet(n_channels=1,
             n_classes=11 if front_or_top == FRONT else 7,
             base_filters=config.base_filters,
             num_layers=config.num_layers,
             expand_factor=config.expand_factor,
             num_groups=config.num_groups,
             dropout_prob=config.dropout_prob).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

os.makedirs("results", exist_ok=True)
os.makedirs(f"results/unet_tds={training_dataset_size}", exist_ok=True)
open(f"results/unet_tds={training_dataset_size}/logs.txt", mode='w')
for epoch in range(config.num_epochs):

    model.train()
    train_losses, train_kypt_dist_losses = [], []
    for batch in tqdm(train_loader, desc=f"Epoch: {epoch+1}/{config.num_epochs} [TRAIN]"):

        images: torch.Tensor = batch["image"].to(device)
        kypt_gt: torch.Tensor = batch["keypoints"].to(device)
        htmp_pred = model(images)

        htmp_size = htmp_pred.shape[2:]
        scale = htmp_pred.shape[-1] / images.shape[-1]
        htmp_gt = generate_heatmaps(kypt_gt, heatmap_size=htmp_size, sigma=sigma, scale=scale)
        loss: torch.Tensor = criterion(htmp_pred, htmp_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_losses.append(loss.item())

        batch_size, num_joints, _, _ = htmp_pred.shape
        kypt_pred = []
        for b in range(batch_size):
            kypt_pred_batch = []
            for i in range(num_joints):
                htmp_pred_batch_joint = htmp_pred[b, i].cpu().detach().numpy()
                y, x = np.unravel_index(np.argmax(htmp_pred_batch_joint), htmp_pred_batch_joint.shape)
                kypt_pred_batch.append([x, y, 2])
            kypt_pred.append(kypt_pred_batch)
        kypt_pred = torch.tensor(kypt_pred, dtype=torch.float32).to(device)
        kypt_dist_loss: torch.Tensor = criterion(kypt_pred, kypt_gt)
        train_kypt_dist_losses.append(kypt_dist_loss.item())

    avg_train_loss = np.average(train_losses)
    avg_train_kypt_dist_loss = np.average(train_kypt_dist_losses)

    model.eval()
    val_losses, val_kypt_dist_losses = [], []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader), desc=f"Epoch: {epoch+1}/{config.num_epochs} [VALID]", total=len(val_loader)):
            images = batch["image"].to(device)
            kypt_gt = batch["keypoints"].to(device)
            htmp_pred = model(images)

            htmp_size = htmp_pred.shape[2:]
            scale = htmp_pred.shape[-1] / images.shape[-1]
            htmp_gt = generate_heatmaps(kypt_gt, heatmap_size=htmp_size, sigma=sigma, scale=scale)
            loss = criterion(htmp_pred, htmp_gt)
            val_losses.append(loss.item())

            batch_size, num_joints, _, _ = htmp_pred.shape
            kypt_pred = []
            for b in range(batch_size):
                kypt_pred_batch = []
                for i in range(num_joints):
                    htmp_pred_batch_joint = htmp_pred[b, i].cpu().detach().numpy()
                    y, x = np.unravel_index(np.argmax(htmp_pred_batch_joint), htmp_pred_batch_joint.shape)
                    kypt_pred_batch.append([x, y, 2])
                kypt_pred.append(kypt_pred_batch)
            kypt_pred = torch.tensor(kypt_pred, dtype=torch.float32, requires_grad=True).to(device)
            kypt_dist_loss: torch.Tensor = criterion(kypt_pred, kypt_gt)
            val_kypt_dist_losses.append(kypt_dist_loss.item())
            
            scale = 1 / scale
            if idx == 0:
                for i in range(images.size(0)):
                    if i == 8: break
                    image = images[i]
                    visualize_heatmaps(image, htmp_pred[i, :1], save_path=f"results/unet_tds={training_dataset_size}/heatmap_{i}_pred.png")
                    visualize_heatmaps(image, htmp_gt[i, :1], save_path=f"results/unet_tds={training_dataset_size}/heatmap_{i}_truth.png")
                    visualize_peaks_from_heatmap(image, htmp_pred[i], save_path=f"results/unet_tds={training_dataset_size}/peaks_{i}_pred.png", scale=scale)
                    visualize_peaks_from_heatmap(image, htmp_gt[i], save_path=f"results/unet_tds={training_dataset_size}/peaks_{i}_truth.png", scale=scale)

    avg_val_loss = np.average(val_losses)
    avg_val_kypt_dist_loss = np.average(val_kypt_dist_losses)

    wandb.log({
        "avg_train_loss": avg_train_loss,
        "avg_train_kypt_dist_loss": avg_train_kypt_dist_loss,
        "avg_val_loss": avg_val_loss,
        "avg_val_kp_dist_loss": avg_val_kypt_dist_loss,
        "learning_rate": get_lr(optimizer)
    })
    print(f"Epoch [{epoch+1}/{config.num_epochs}]\n" + \
          f"Training   Loss: {avg_train_loss:.8f}, Training   Keypoint Distance Loss: {avg_train_kypt_dist_loss:.8f}\n" + \
          f"Validation Loss: {avg_val_loss:.8f}, Validation Keypoint Distance Loss: {avg_val_kypt_dist_loss:.8f}\n" + \
          f"LR: {get_lr(optimizer):.10f}")

    with open(f"results/unet_tds={training_dataset_size}/logs.txt", mode='a') as log_file:
        log_file.write(f"{epoch},{avg_train_loss},{avg_train_kypt_dist_loss},{avg_val_loss},{avg_val_kypt_dist_loss},{get_lr(optimizer)}\n")
