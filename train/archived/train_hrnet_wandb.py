import wandb
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from train.utils import get_lr, generate_heatmaps
from models.hrnet import PoseHighResolutionNet
from data.dataset import Resize, ToTensor, RandomHorizontalFlip, RandomRotate90Degree, CroppedKeypointsDataset
from torch.utils.data import random_split, DataLoader



HEATMAP = "heatmap"
PEAK = "peak"
FRONT = "front"
TOP = "top"

heatmap_or_peak = HEATMAP
front_or_top = FRONT
sigma = 3

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "avg_val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "batch_size": {
            "values": [ 4, 8 ]
        },
        "gamma": {
            "distribution": "uniform", "min": 0.9998, "max": 0.99999
        },
        "learning_rate": {
            "distribution": "uniform", "min": 1e-4, "max": 1e-3
        },
        "num_epochs": {
            "value": 50
        },
        "num_block": {
            "values": [ 1, 2, 3 ]
        },
        "num_channel_base": {
            "values": [ 16, 24, 32, 48 ]
        },
        "chnl_exp_factor": {
            "values": [ 1.5, 2, 2.5, 3 ]
        }
    }
}

#                                              TDPE = Top-Down Pose Estimation
sweep_id = wandb.sweep(sweep_config, project=f"TDPE_HRNet_{front_or_top}_sigma-{sigma}")

def train(config=None):

    torch.manual_seed(42)

    with wandb.init(config=config):
        config = wandb.config
        
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
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size*2, shuffle=False)

        cfg = {
            "NUM_JOINTS": 11 if front_or_top == FRONT else 7,
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "STAGE2": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 2,
                    "BLOCK": "BASIC",
                    "NUM_BLOCKS": [ config.num_block for _ in range(2) ],
                    "NUM_CHANNELS": [ int(config.num_channel_base*(config.chnl_exp_factor**i)) for i in range(2) ],
                    "FUSE_METHOD": "SUM"
                },
                "STAGE3": {
                    "NUM_MODULES": 4,
                    "NUM_BRANCHES": 3,
                    "BLOCK": "BASIC",
                    "NUM_BLOCKS": [ config.num_block for _ in range(3) ],
                    "NUM_CHANNELS": [ int(config.num_channel_base*(config.chnl_exp_factor**i)) for i in range(3) ],
                    "FUSE_METHOD": "SUM"
                },
                "STAGE4": {
                    "NUM_MODULES": 3,
                    "NUM_BRANCHES": 4,
                    "BLOCK": "BASIC",
                    "NUM_BLOCKS": [ config.num_block for _ in range(4) ],
                    "NUM_CHANNELS": [ int(config.num_channel_base*(config.chnl_exp_factor**i)) for i in range(4) ],
                    "FUSE_METHOD": "SUM"
                }
            }
        }

        model = PoseHighResolutionNet(cfg).to(device)
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

                if heatmap_or_peak == "heatmap":
                    heatmap_size = outputs.shape[2:]
                    scale = outputs.shape[-1] / images.shape[-1]
                    keypoints_heatmaps = generate_heatmaps(keypoints, heatmap_size=heatmap_size, sigma=sigma, scale=scale)
                    loss: torch.Tensor = criterion(outputs, keypoints_heatmaps)

                elif heatmap_or_peak == "peak":
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

                    if heatmap_or_peak == "heatmap":
                        heatmap_size = outputs.shape[2:]
                        scale = outputs.shape[-1] / images.shape[-1]
                        keypoints_heatmaps = generate_heatmaps(keypoints, heatmap_size=heatmap_size, sigma=sigma, scale=scale)
                        loss = criterion(outputs, keypoints_heatmaps)

                    elif heatmap_or_peak == "peak":
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
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "learning_rate": get_lr(optimizer)
            })
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.8f}, " + \
                  f"Validation Loss: {avg_val_loss:.8f}, LR: {get_lr(optimizer):.10f}")
            
            if epoch == 20:
                if avg_val_loss > 0.006:
                    break

wandb.agent(sweep_id, function=train)
