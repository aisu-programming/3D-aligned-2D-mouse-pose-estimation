import os
import torch
torch.manual_seed(42)
import wandb
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from train.utils import get_lr, generate_heatmaps, visualize_heatmaps, visualize_peaks_from_heatmap, \
    NTXentLoss
from models.unet_embedding_head import UNet
from data.dataset import PairedResize, PairedToTensor, PairedRandomHorizontalFlip, PairedRandomRotate90Degree, PairedCroppedKeypointsDataset
from torch.utils.data import Subset, DataLoader



sigma = 12
training_dataset_size = 20

wandb.init(
    project="CL_UNet_20",
    name=f"CL_UNet_tds={training_dataset_size}",
    config={
        "base_filters": 16,
        "batch_size": 16,
        "dropout_prob": 0.01,
        "expand_factor": 2,
        "gamma": 0.99991,
        "learning_rate": 0.006,
        "num_groups": 8,
        "num_layers": 6,
        "num_epochs": 150,
        "temperature": 0.6,
        "align_loss_factor": 0.01,
        "training_dataset_size": training_dataset_size,
    }
)

config = wandb.config

transform = transforms.Compose([
    PairedResize(256),
    PairedRandomHorizontalFlip(),
    PairedRandomRotate90Degree(),
    PairedToTensor(),
])
dataset = PairedCroppedKeypointsDataset(
    annotations_file_front="datasets/MARS/MARS_cropped_keypoints_front_YOLO.json",
    img_dir_front="datasets/MARS/cropped_images_front_YOLO",
    annotations_file_top="datasets/MARS/MARS_cropped_keypoints_top_YOLO.json",
    img_dir_top="datasets/MARS/cropped_images_top_YOLO",
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
             n_classes=11,
             base_filters=config.base_filters,
             num_layers=config.num_layers,
             expand_factor=config.expand_factor,
             num_groups=config.num_groups,
             dropout_prob=config.dropout_prob).to(device)
mse_loss_fn = torch.nn.MSELoss()
ntxent_loss_fn = NTXentLoss(temperature=config.temperature, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

os.makedirs("results", exist_ok=True)
os.makedirs(f"results/cl_unet_tds={training_dataset_size}", exist_ok=True)
open(f"results/cl_unet_tds={training_dataset_size}/logs.txt", mode='w')
for epoch in range(config.num_epochs):

    model.train()
    train_losses_front, train_losses_top, train_losses_align = [], [], []
    train_kypt_dist_losses_front, train_kypt_dist_losses_top = [], []

    for batch in tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{config.num_epochs} [TRAIN]"):

        images, kypt_gt, kypt_pred, htmp_pred, htmp_gt, reps = {}, {}, {}, {}, {}, {}
        for view in ["front", "top"]:

            images[view] = batch["image"][view].to(device)
            kypt_gt[view] = batch["keypoints"][view].to(device)
            htmp_pred[view], reps[view] = model(images[view], ret_rep=True)
            if view == "top":
                htmp_pred[view] = htmp_pred[view][:, :7, :, :]
            htmp_size = htmp_pred[view].shape[2:]
            scale = htmp_pred[view].shape[-1] / images[view].shape[-1]
            htmp_gt[view] = generate_heatmaps(kypt_gt[view], heatmap_size=htmp_size, sigma=sigma, scale=scale)

            batch_size, num_joints, _, _ = htmp_pred[view].shape
            kypt_pred[view] = []
            for b in range(batch_size):
                kypt_pred_batch = []
                for i in range(num_joints):
                    htmp_pred_batch_joint = htmp_pred[view][b, i].cpu().detach().numpy()
                    y, x = np.unravel_index(np.argmax(htmp_pred_batch_joint), htmp_pred_batch_joint.shape)
                    kypt_pred_batch.append([x, y, 2])
                kypt_pred[view].append(kypt_pred_batch)
            kypt_pred[view] = torch.tensor(kypt_pred[view], dtype=torch.float32, requires_grad=True).to(device)

        loss_front: torch.Tensor = mse_loss_fn(htmp_pred["front"], htmp_gt["front"])
        loss_top: torch.Tensor = mse_loss_fn(htmp_pred["top"], htmp_gt["top"])
        reps_front = [reps["front"]]
        reps_top = [reps["top"]]
        loss_align: torch.Tensor = ntxent_loss_fn(reps_front, reps_top)
        # Multi-View Consistency Loss
        keypoints_front = kypt_pred["front"]
        keypoints_top = kypt_pred["top"]

        # Align keypoints
        keypoint_mapping = [0, 1, 2, 3, 4, 5, 6]  # Indices of relevant keypoints in the front view

        multi_view_loss = multi_view_consistency_loss(kypt_pred["front"][:, keypoint_mapping, :], kypt_pred["top"])

        # Total loss
        loss_total = (
                loss_front +
                loss_top +
                loss_align * config.align_loss_factor +
                multi_view_loss * 0.005  # Weight for multi-view consistency loss
        )
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        lr_scheduler.step()

        train_losses_front.append(loss_front.item())
        train_losses_top.append(loss_top.item())
        train_losses_align.append(loss_align.item())

        kypt_dist_loss_front: torch.Tensor = mse_loss_fn(kypt_pred["front"], kypt_gt["front"])
        kypt_dist_loss_top: torch.Tensor = mse_loss_fn(kypt_pred["top"], kypt_gt["top"])
        train_kypt_dist_losses_front.append(kypt_dist_loss_front.item())
        train_kypt_dist_losses_top.append(kypt_dist_loss_top.item())

    avg_train_loss_front = np.average(train_losses_front)
    avg_train_loss_top = np.average(train_losses_top)
    avg_train_loss_align = np.average(train_losses_align) * config.align_loss_factor

    avg_train_kypt_dist_loss_front = np.average(train_kypt_dist_losses_front)
    avg_train_kypt_dist_loss_top = np.average(train_kypt_dist_losses_top)

    model.eval()
    val_losses_front, val_losses_top, val_losses_align = [], [], []
    val_kypt_dist_losses_front, val_kypt_dist_losses_top = [], []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader),
                               desc=f"Epoch: {epoch + 1}/{config.num_epochs} [VALID]",
                               total=len(val_loader)):

            images, kypt_gt, kypt_pred, htmp_pred, htmp_gt, reps = {}, {}, {}, {}, {}, {}
            for view in ["front", "top"]:

                images[view] = batch["image"][view].to(device)
                kypt_gt[view] = batch["keypoints"][view].to(device)
                htmp_pred[view], reps[view] = model(images[view], ret_rep=True)
                if view == "top":
                    htmp_pred[view] = htmp_pred[view][:, :7, :, :]
                htmp_size = htmp_pred[view].shape[2:]
                scale = htmp_pred[view].shape[-1] / images[view].shape[-1]
                htmp_gt[view] = generate_heatmaps(kypt_gt[view], heatmap_size=htmp_size, sigma=sigma, scale=scale)

                batch_size, num_joints, _, _ = htmp_pred[view].shape
                kypt_pred[view] = []
                for b in range(batch_size):
                    kypt_pred_batch = []
                    for i in range(num_joints):
                        htmp_pred_batch_joint = htmp_pred[view][b, i].cpu().detach().numpy()
                        y, x = np.unravel_index(np.argmax(htmp_pred_batch_joint), htmp_pred_batch_joint.shape)
                        kypt_pred_batch.append([x, y, 2])
                    kypt_pred[view].append(kypt_pred_batch)
                kypt_pred[view] = torch.tensor(kypt_pred[view], dtype=torch.float32, requires_grad=True).to(device)

            loss_front: torch.Tensor = mse_loss_fn(htmp_pred["front"], htmp_gt["front"])
            loss_top: torch.Tensor = mse_loss_fn(htmp_pred["top"], htmp_gt["top"])
            reps_front = [reps["front"]]
            reps_top = [reps["top"]]
            loss_align: torch.Tensor = ntxent_loss_fn(reps_front, reps_top)

            val_losses_front.append(loss_front.item())
            val_losses_top.append(loss_top.item())
            val_losses_align.append(loss_align.item())

            kypt_dist_loss_front: torch.Tensor = mse_loss_fn(kypt_pred["front"], kypt_gt["front"])
            kypt_dist_loss_top: torch.Tensor = mse_loss_fn(kypt_pred["top"], kypt_gt["top"])
            val_kypt_dist_losses_front.append(kypt_dist_loss_front.item())
            val_kypt_dist_losses_top.append(kypt_dist_loss_top.item())

            scale = 1 / scale
            if idx == 0:
                for view in ["front", "top"]:
                    for i in range(images[view].size(0)):
                        if i == 8: break
                        image = images[view][i]
                        visualize_heatmaps(image, htmp_pred[view][i, :1],
                                           save_path=f"results/cl_unet_tds={training_dataset_size}/heatmap_{i}_{view}_pred.png")
                        visualize_heatmaps(image, htmp_gt[view][i, :1],
                                           save_path=f"results/cl_unet_tds={training_dataset_size}/heatmap_{i}_{view}_truth.png")
                        visualize_peaks_from_heatmap(image, htmp_pred[view][i],
                                                     save_path=f"results/cl_unet_tds={training_dataset_size}/peaks_{i}_{view}_pred.png",
                                                     scale=scale)
                        visualize_peaks_from_heatmap(image, htmp_gt[view][i],
                                                     save_path=f"results/cl_unet_tds={training_dataset_size}/peaks_{i}_{view}_truth.png",
                                                     scale=scale)

    avg_val_loss_front = np.average(val_losses_front)
    avg_val_loss_top = np.average(val_losses_top)
    avg_val_loss_align = np.average(val_losses_align) * config.align_loss_factor

    avg_val_kypt_dist_loss_front = np.average(val_kypt_dist_losses_front)
    avg_val_kypt_dist_loss_top = np.average(val_kypt_dist_losses_top)

    wandb.log({
        "avg_train_loss_front": avg_train_loss_front,
        "avg_train_loss_top": avg_train_loss_top,
        "avg_train_loss_align": avg_train_loss_align,
        "avg_train_kypt_dist_loss_front": avg_train_kypt_dist_loss_front,
        "avg_train_kypt_dist_loss_top": avg_train_kypt_dist_loss_top,
        "avg_val_loss_front": avg_val_loss_front,
        "avg_val_loss_top": avg_val_loss_top,
        "avg_val_loss_align": avg_val_loss_align,
        "avg_val_kypt_dist_loss_front": avg_val_kypt_dist_loss_front,
        "avg_val_kypt_dist_loss_top": avg_val_kypt_dist_loss_top,
        "learning_rate": get_lr(optimizer),
    })
    print(f"Epoch [{epoch + 1}/{config.num_epochs}]:\n" + \
          f"Training Loss Front: {avg_train_loss_front:.8f}, Training   Loss Top: {avg_train_loss_top:.8f}, Training   Loss Align: {avg_train_loss_align:.8f}\n" + \
          f"Training Keypoint Distance Loss Front: {avg_train_kypt_dist_loss_front:.8f}, Training Keypoint Distance Loss Top: {avg_train_kypt_dist_loss_top:.8f}\n" + \
          f"Validation Loss Front: {avg_val_loss_front:.8f}, Validation Loss Top: {avg_val_loss_top:.8f}, Validation Loss Align: {avg_val_loss_align:.8f}\n" + \
          f"Validation Keypoint Distance Loss Front: {avg_val_kypt_dist_loss_front:.8f}, Validation Keypoint Distance Loss Top: {avg_val_kypt_dist_loss_top:.8f}\n" + \
          f"LR: {get_lr(optimizer):.10f}\n")

    with open(f"results/cl_unet_tds={training_dataset_size}/logs.txt", mode='a') as log_file:
        log_file.write(f"{epoch},{avg_train_loss_front},{avg_train_loss_top},{avg_train_loss_align}," + \
                       f"{avg_train_kypt_dist_loss_front},{avg_train_kypt_dist_loss_top}," + \
                       f"{avg_val_loss_front},{avg_val_loss_top},{avg_val_loss_align}," + \
                       f"{avg_val_kypt_dist_loss_front},{avg_val_kypt_dist_loss_top},{get_lr(optimizer)}\n")