import wandb
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from train.utils import InfoNCELoss, get_lr, generate_heatmaps
from models.unet import UNet
from data.dataset import PairedResize, PairedToTensor, PairedRandomHorizontalFlip, PairedRandomRotate90Degree, PairedCroppedKeypointsDataset
from torch.utils.data import random_split, DataLoader



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
        },
        "temperature": {
            "distribution": "uniform", "min": 0.1, "max": 0.9
        },
        "align_loss_factor": {
            "distribution": "uniform", "min": 1e-3, "max": 0.1
        }
    }
}

#                                              TDPE = Top-Down Pose Estimation
sweep_id = wandb.sweep(sweep_config, project=f"TDPE_CL_UNet_sigma-{sigma}")

def train(config=None):

    torch.manual_seed(42)

    with wandb.init(config=config):
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

        dataset_size = len(dataset)
        train_ratio = 0.8
        train_size = int(train_ratio * dataset_size)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
        info_nce_loss_fn = InfoNCELoss(temperature=config.temperature, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

        for epoch in range(config.num_epochs):

            model.train()
            train_losses_front, train_losses_top, train_losses_align = [], [], []
            # train_kypt_dist_losses_front, train_kypt_dist_losses_top = [], []
            for batch in tqdm(train_loader, desc=f"Epoch: {epoch+1}/{config.num_epochs} [TRAIN]"):

                images, kypt_gt, kypt_pred, htmp_pred, htmp_gt, reps = {}, {}, {}, {}, {}, {}
                for view in [ "front", "top" ]:

                    images[view] = batch["image"][view].to(device)
                    kypt_gt[view] = batch["keypoints"][view].to(device)
                    htmp_pred[view], reps[view] = model(images[view], ret_rep=True)
                    if view == "top":
                        htmp_pred[view] = htmp_pred[view][:, :7, :, :]
                    htmp_size = htmp_pred[view].shape[2:]
                    scale = htmp_pred[view].shape[-1] / images[view].shape[-1]
                    htmp_gt[view] = generate_heatmaps(kypt_gt[view], heatmap_size=htmp_size, sigma=sigma, scale=scale)

                    # batch_size, num_joints, _, _ = htmp_pred[view].shape
                    # kypt_pred[view] = []
                    # for b in range(batch_size):
                    #     kypt_pred_batch = []
                    #     for i in range(num_joints):
                    #         htmp_pred_batch_joint = htmp_pred[view][b, i].cpu().detach().numpy()
                    #         y, x = np.unravel_index(np.argmax(htmp_pred_batch_joint), htmp_pred_batch_joint.shape)
                    #         kypt_pred_batch.append([x, y, 2])
                    #     kypt_pred[view].append(kypt_pred_batch)
                    # kypt_pred[view] = torch.tensor(kypt_pred[view], dtype=torch.float32, requires_grad=True).to(device)
                
                loss_front: torch.Tensor = mse_loss_fn(htmp_pred["front"], htmp_gt["front"])
                loss_top:   torch.Tensor = mse_loss_fn(htmp_pred["top"], htmp_gt["top"])

                reps_front = [ reps["front"] ]
                reps_top   = [ reps["top"]   ]
                loss_align: torch.Tensor = info_nce_loss_fn(reps_front, reps_top)
                loss_total: torch.Tensor = loss_front + loss_top + loss_align * config.align_loss_factor

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                lr_scheduler.step()

                train_losses_front.append(loss_front.item())
                train_losses_top.append(loss_top.item())
                train_losses_align.append(loss_align.item())

                # kypt_dist_loss_front: torch.Tensor = mse_loss_fn(kypt_pred["front"], kypt_gt["front"])
                # kypt_dist_loss_top:   torch.Tensor = mse_loss_fn(kypt_pred["top"], kypt_gt["top"])
                # train_kypt_dist_losses_front.append(kypt_dist_loss_front.item())
                # train_kypt_dist_losses_top.append(kypt_dist_loss_top.item())

            avg_train_loss_front = np.average(train_losses_front)
            avg_train_loss_top   = np.average(train_losses_top)
            avg_train_loss_align = np.average(train_losses_align)

            # avg_train_kypt_dist_loss_front = np.average(train_kypt_dist_losses_front)
            # avg_train_kypt_dist_loss_top   = np.average(train_kypt_dist_losses_top)



            model.eval()
            val_losses_front, val_losses_top, val_losses_align = [], [], []
            # val_kypt_dist_losses_front, val_kypt_dist_losses_top = [], []
            with torch.no_grad():
                for _, batch in tqdm(enumerate(val_loader),
                                     desc=f"Epoch: {epoch+1}/{config.num_epochs} [VALID]",
                                     total=len(val_loader)):

                    images, kypt_gt, kypt_pred, htmp_pred, htmp_gt, reps = {}, {}, {}, {}, {}, {}
                    for view in [ "front", "top" ]:

                        images[view] = batch["image"][view].to(device)
                        kypt_gt[view] = batch["keypoints"][view].to(device)
                        htmp_pred[view], reps[view] = model(images[view], ret_rep=True)
                        if view == "top":
                            htmp_pred[view] = htmp_pred[view][:, :7, :, :]
                        htmp_size = htmp_pred[view].shape[2:]
                        scale = htmp_pred[view].shape[-1] / images[view].shape[-1]
                        htmp_gt[view] = generate_heatmaps(kypt_gt[view], heatmap_size=htmp_size, sigma=sigma, scale=scale)

                        # batch_size, num_joints, _, _ = htmp_pred[view].shape
                        # kypt_pred[view] = []
                        # for b in range(batch_size):
                        #     kypt_pred_batch = []
                        #     for i in range(num_joints):
                        #         htmp_pred_batch_joint = htmp_pred[view][b, i].cpu().detach().numpy()
                        #         y, x = np.unravel_index(np.argmax(htmp_pred_batch_joint), htmp_pred_batch_joint.shape)
                        #         kypt_pred_batch.append([x, y, 2])
                        #     kypt_pred[view].append(kypt_pred_batch)
                        # kypt_pred[view] = torch.tensor(kypt_pred[view], dtype=torch.float32, requires_grad=True).to(device)
                
                    loss_front: torch.Tensor = mse_loss_fn(htmp_pred["front"], htmp_gt["front"])
                    loss_top:   torch.Tensor = mse_loss_fn(htmp_pred["top"], htmp_gt["top"])
                    
                    reps_front = [ reps["front"] ]
                    reps_top   = [ reps["top"]   ]
                    loss_align: torch.Tensor = info_nce_loss_fn(reps_front, reps_top)

                    val_losses_front.append(loss_front.item())
                    val_losses_top.append(loss_top.item())
                    val_losses_align.append(loss_align.item())

                    # kypt_dist_loss_front: torch.Tensor = mse_loss_fn(kypt_pred["front"], kypt_gt["front"])
                    # kypt_dist_loss_top:   torch.Tensor = mse_loss_fn(kypt_pred["top"],   kypt_gt["top"])
                    # val_kypt_dist_losses_front.append(kypt_dist_loss_front.item())
                    # val_kypt_dist_losses_top.append(kypt_dist_loss_top.item())

            avg_val_loss_front = np.average(val_losses_front)
            avg_val_loss_top   = np.average(val_losses_top)
            avg_val_loss_align = np.average(val_losses_align)

            # avg_val_kypt_dist_loss_front = np.average(val_kypt_dist_losses_front)
            # avg_val_kypt_dist_loss_top   = np.average(val_kypt_dist_losses_top)



            wandb.log({
                "avg_train_loss_front": avg_train_loss_front,
                "avg_train_loss_top"  : avg_train_loss_top,
                "avg_train_loss_align": avg_train_loss_align,
                # "avg_train_kypt_dist_loss_front": avg_train_kypt_dist_loss_front,
                # "avg_train_kypt_dist_loss_top"  : avg_train_kypt_dist_loss_top,
                "avg_val_loss_front": avg_val_loss_front,
                "avg_val_loss_top"  : avg_val_loss_top,
                "avg_val_loss_align": avg_val_loss_align,
                # "avg_val_kypt_dist_loss_front": avg_val_kypt_dist_loss_front,
                # "avg_val_kypt_dist_loss_top"  : avg_val_kypt_dist_loss_top,
                "learning_rate": get_lr(optimizer)
            })
            print(f"Epoch [{epoch+1}/{config.num_epochs}]:\n" + \
                  f"Training   Loss Front: {avg_train_loss_front:.8f}, Training   Loss Top: {avg_train_loss_top:.8f}, Training   Loss Align: {avg_train_loss_align:.8f}\n" + \
                  f"Validation Loss Front: {avg_val_loss_front:.8f}, Validation Loss Top: {avg_val_loss_top:.8f}, Validation Loss Align: {avg_val_loss_align:.8f}\n" + \
                  f"LR: {get_lr(optimizer):.10f}\n")
            
            if epoch == 20:
                if avg_val_loss_front > 0.0055:
                    break

wandb.agent(sweep_id, function=train)
