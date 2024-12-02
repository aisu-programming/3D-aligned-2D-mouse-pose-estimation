import wandb
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from train.utils import get_lr, generate_heatmaps
from models.unet import UNet
from data.dataset import Resize, Pad, ToTensor, RandomHorizontalFlip, RandomRotate90Degree, KeypointsDataset, CroppedKeypointsDataset
from torch.utils.data import random_split, DataLoader



HEATMAP_OR_PEAK = "heatmap"
TOP_OR_FRONT = "front"
SIGMA = 30

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "avg_val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "base_filters": {
            "values": [ 16, 24, 32, 48 ]
        },
        "batch_size": {
            "values": [ 8, 16, 32, 64 ]
        },
        "dropout_prob": {
            "values": [ 0.05, 0.1, 0.2 ]
        },
        "expand_factor": {
            "values": [ 1.5, 2, 2.5 ]
        },
        "gamma": {
            "distribution": "uniform", "min": 0.998, "max": 0.99999
        },
        "learning_rate": {
            "distribution": "uniform", "min": 5e-5, "max": 5e-3
        },
        "num_groups": {
            "values": [ 4, 8, 16 ]
        },
        "num_layers": {
            "values": [ 4, 5, 6 ]
        },
        "num_epochs": {
            "value": 50
        }
    }
}

#                                              TDPE = Top-Down Pose Estimation
sweep_id = wandb.sweep(sweep_config, project=f"TDPE_{TOP_OR_FRONT}_heatmap_sigma-{SIGMA}")

def train(config=None):

    with wandb.init(config=config):
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

        if TOP_OR_FRONT == "top":
            transform = transforms.Compose([
                Resize((180, 180)),
                RandomHorizontalFlip(),
                RandomRotate90Degree(),
                ToTensor(),
            ])
        elif TOP_OR_FRONT == "front":
            transform = transforms.Compose([
                Resize((240, 240)),
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
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False)

        model = UNet(n_channels=1,
                     n_classes=9 if TOP_OR_FRONT == "top" else 11,
                     base_filters=config.base_filters,
                     num_layers=config.num_layers,
                     expand_factor=config.expand_factor,
                     num_groups=config.num_groups,
                     dropout_prob=config.dropout_prob).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

        num_epochs = config.num_epochs
        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            for batch in tqdm(train_loader, desc=f"Epoch: {epoch+1}/{num_epochs} [TRAIN]"):
                images = batch["image"].to(device)
                keypoints = batch["keypoints"].to(device)
                outputs = model(images)

                if HEATMAP_OR_PEAK == "heatmap":
                    keypoints_heatmaps = generate_heatmaps(keypoints, heatmap_size=outputs.shape[2:], sigma=SIGMA)
                    loss: torch.Tensor = criterion(outputs, keypoints_heatmaps)
                elif HEATMAP_OR_PEAK == "peak":
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
                else:
                    raise Exception

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_losses.append(loss.item())
            avg_train_loss = np.average(train_losses)

            model.eval()
            val_losses = []
            with torch.no_grad():
                for _, batch in tqdm(enumerate(val_loader),
                                     desc=f"Epoch: {epoch+1}/{num_epochs} [VALID]",
                                     total=len(val_loader)):
                    images = batch["image"].to(device)
                    keypoints = batch["keypoints"].to(device)
                    outputs = model(images)

                    if HEATMAP_OR_PEAK == "heatmap":
                        keypoints_heatmaps = generate_heatmaps(keypoints, heatmap_size=outputs.shape[2:], sigma=SIGMA)
                        loss = criterion(outputs, keypoints_heatmaps)
                    elif HEATMAP_OR_PEAK == "peak":
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
                    else:
                        raise Exception

                    val_losses.append(loss.item())
            avg_val_loss = np.average(val_losses)

            wandb.log({
                # "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "learning_rate": get_lr(optimizer)
            })

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.8f}, " + \
                  f"Validation Loss: {avg_val_loss:.8f}, LR: {optimizer.param_groups[0]['lr']:.10f}")
            
            # if epoch == 20:
            #     if avg_val_loss > 0.012:
            #         break

wandb.agent(sweep_id, function=train)
