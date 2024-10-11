import wandb
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from utils import get_lr, generate_heatmaps
from models.unet import UNet
from data.dataset import Resize, Pad, ToTensor, RandomHorizontalFlip, CocoKeypointsDataset
from torch.utils.data import random_split, DataLoader



sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'avg_val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'uniform', 'min': 1e-4, 'max': 1e-2
        },
        'gamma': {
            'distribution': 'uniform', 'min': 0.99, 'max': 0.9999
        },
        'batch_size': {
            'values': [ 8, 16, 32, 64 ]
        },
        'base_filters': {
            'values': [ 16, 24, 32, 48, 64 ]
        },
        'num_layers': {
            'values': [ 3, 4, 5, 6, 7 ]
        },
        'expand_factor': {
            'values': [ 1.5, 2, 2.5 ]
        },
        'num_epochs': {
            'value': 50
        },
        'num_groups': {
            'values': [ 4, 8, 16 ]
        },
        'dropout_prob': {
            'values': [ 0.1, 0.2, 0.3 ]
        },
        "sigma": {
            "values": [ 3, 5, 10, 15 ]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="Pose Estimation")

def train(config=None):

    with wandb.init(config=config):
        config = wandb.config

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
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False)

        model = UNet(n_channels=1, n_classes=18,
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
                keypoints_heatmaps = generate_heatmaps(keypoints, heatmap_size=outputs.shape[2:], sigma=config.sigma)
                loss = criterion(outputs, keypoints_heatmaps)
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
                    keypoints_heatmaps = generate_heatmaps(keypoints, heatmap_size=outputs.shape[2:], sigma=config.sigma)
                    loss = criterion(outputs, keypoints_heatmaps)
                    val_losses.append(loss.item())
            avg_val_loss = np.average(val_losses)

            wandb.log({
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'learning_rate': get_lr(optimizer)
            })

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.8f}, " + \
                  f"Validation Loss: {avg_val_loss:.8f}, LR: {optimizer.param_groups[0]['lr']:.10f}")

wandb.agent(sweep_id, function=train)
