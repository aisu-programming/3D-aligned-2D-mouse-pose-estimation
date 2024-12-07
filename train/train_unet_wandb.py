import wandb
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from train.utils import get_lr, generate_heatmaps
from models.unet import UNet
from data.dataset import Resize, Pad, ToTensor, RandomHorizontalFlip, RandomRotate90Degree, KeypointsDataset, CroppedKeypointsDataset
from torch.utils.data import random_split, DataLoader



FRONT = "front"
TOP = "top"

front_or_top = FRONT
sigma = 12

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "avg_val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "base_filters": {
            "values": [ 4, 8, 16, 24 ]
        },
        "batch_size": {
            "values": [ 8, 16, 32 ]
        },
        "dropout_prob": {
            "values": [ 0, 0.02, 0.05 ]
        },
        "expand_factor": {
            "values": [ 1.5, 2, 2.5 ]
        },
        "gamma": {
            "distribution": "uniform", "min": 0.9998, "max": 0.999999
        },
        "learning_rate": {
            "distribution": "uniform", "min": 1e-3, "max": 1e-2
        },
        "num_groups": {
            "values": [ 4, 8, 16 ]
        },
        "num_layers": {
            "values": [ 5, 6, 7, 8 ]
        },
        "num_epochs": {
            "value": 50
        }
    }
}

#                                              TDPE = Top-Down Pose Estimation
sweep_id = wandb.sweep(sweep_config, project=f"TDPE_UNet_{front_or_top}_sigma-{sigma}")

def train(config=None):

    torch.manual_seed(42)

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

        num_epochs = config.num_epochs
        for epoch in range(num_epochs):
            model.train()
            train_losses, train_kypt_dist_losses = [], []
            for batch in tqdm(train_loader, desc=f"Epoch: {epoch+1}/{num_epochs} [TRAIN]"):
                images = batch["image"].to(device)
                kypt_gt = batch["keypoints"].to(device)
                htmp_pred = model(images)

                htmp_size = htmp_pred.shape[2:]
                scale = htmp_pred.shape[-1] / images.shape[-1]
                htmp_gt = generate_heatmaps(kypt_gt, heatmap_size=htmp_size, sigma=sigma, scale=scale)
                loss: torch.Tensor = criterion(htmp_pred, htmp_gt)

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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_losses.append(loss.item())
                train_kypt_dist_losses.append(kypt_dist_loss.item())

            avg_train_loss = np.average(train_losses)
            avg_train_kypt_dist_loss = np.average(train_kypt_dist_losses)

            model.eval()
            val_losses, val_kypt_dist_losses = [], []
            with torch.no_grad():
                for _, batch in tqdm(enumerate(val_loader),
                                     desc=f"Epoch: {epoch+1}/{num_epochs} [VALID]",
                                     total=len(val_loader)):
                    images = batch["image"].to(device)
                    kypt_gt = batch["keypoints"].to(device)
                    htmp_pred = model(images)

                    htmp_size = htmp_pred.shape[2:]
                    scale = htmp_pred.shape[-1] / images.shape[-1]
                    htmp_gt = generate_heatmaps(kypt_gt, heatmap_size=htmp_size, sigma=sigma, scale=scale)
                    loss = criterion(htmp_pred, htmp_gt)

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
                    kypt_dist_loss = criterion(kypt_pred, kypt_gt)
                        
                    val_losses.append(loss.item())
                    val_kypt_dist_losses.append(kypt_dist_loss.item())

            avg_val_loss = np.average(val_losses)
            avg_val_kypt_dist_loss = np.average(val_kypt_dist_losses)

            wandb.log({
                "avg_train_loss": avg_train_loss,
                "avg_train_kypt_dist_loss": avg_train_kypt_dist_loss,
                "avg_val_loss": avg_val_loss,
                "avg_val_kp_dist_loss": avg_val_kypt_dist_loss,
                "learning_rate": get_lr(optimizer)
            })
            print(f"Epoch [{epoch+1}/{num_epochs}]\n" + \
                  f"Training   Heatmap Loss: {avg_train_loss:.8f}, Training   Keypoint Distance Loss: {avg_train_kypt_dist_loss:.8f}\n" + \
                  f"Validation Heatmap Loss: {avg_val_loss:.8f}, Validation Keypoint Distance Loss: {avg_val_kypt_dist_loss:.8f}\n" + \
                  f"LR: {get_lr(optimizer):.10f}")
            
            if epoch == 20:
                if avg_val_loss > 0.0063:
                    break

wandb.agent(sweep_id, function=train)
