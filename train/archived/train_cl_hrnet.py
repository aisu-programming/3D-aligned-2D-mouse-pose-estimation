import os
import torch
torch.manual_seed(42)
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from train.utils import InfoNCELoss, get_lr, generate_heatmaps, visualize_peaks_from_heatmap, visualize_heatmaps
from models.hrnet import PoseHighResolutionNet
from data.dataset import PairedResize, PairedRandomHorizontalFlip, PairedRandomRotate90Degree, PairedToTensor, PairedCroppedKeypointsDataset
from torch.utils.data import random_split, DataLoader



sigma = 3
batch_size = 8
gamma = 0.99994
learning_rate = 0.005
num_epochs = 200
num_block = 3
num_chnl_base = 24
chnl_exp_factor = 2
temperature = 0.7
align_loss_factor = 0.08



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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

cfg = {
    "NUM_JOINTS": 11,
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
heatmap_loss_fn = torch.nn.MSELoss()
contrastive_loss_fn = InfoNCELoss(temperature=temperature, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

os.makedirs("results", exist_ok=True)
os.makedirs("results/cl_hrnet", exist_ok=True)
for epoch in range(num_epochs):
    model.train()
    train_losses_front, train_losses_top, train_losses_align = [], [], []
    for batch in tqdm(train_loader, desc=f"Epoch: {epoch+1}/{num_epochs} [TRAIN]"):

        images, kp_gt, kp_htmp_pred, kp_htmp_gt, reps = {}, {}, {}, {}, {}
        for view in [ "front", "top" ]:
            images[view] = batch["image"][view].to(device)
            kp_gt[view] = batch["keypoints"][view].to(device)
            kp_htmp_pred[view], reps[view] = model(images[view], ret_rep=True)
            if view == "top":
                kp_htmp_pred[view] = kp_htmp_pred[view][:, :7, :, :]
            htmp_size = kp_htmp_pred[view].shape[2:]
            scale = kp_htmp_pred[view].shape[-1] / images[view].shape[-1]
            kp_htmp_gt[view] = generate_heatmaps(kp_gt[view], heatmap_size=htmp_size, sigma=sigma, scale=scale)
        
        loss_front: torch.Tensor = heatmap_loss_fn(kp_htmp_pred["front"], kp_htmp_gt["front"])
        loss_top:   torch.Tensor = heatmap_loss_fn(kp_htmp_pred["top"], kp_htmp_gt["top"])

        reps_front, reps_top = reps["front"][1], reps["top"][1]
        reps_front = [ torch.nn.functional.avg_pool3d(rep, 8) for rep in reps_front ]
        reps_top   = [ torch.nn.functional.avg_pool3d(rep, 8) for rep in reps_top ]
        # reps_front = [ torch.nn.functional.adaptive_avg_pool2d(rep, (rep.size(0), 512)) for rep in reps_front ]
        # reps_top   = [ torch.nn.functional.adaptive_avg_pool2d(rep, (rep.size(0), 512)) for rep in reps_top ]
        loss_align: torch.Tensor = contrastive_loss_fn(reps_front, reps_top)

        loss_total = loss_front + loss_top + loss_align * align_loss_factor
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        lr_scheduler.step()

        train_losses_front.append(loss_front.item())
        train_losses_top.append(loss_top.item())
        train_losses_align.append(loss_align.item())

    avg_train_loss_front = np.average(train_losses_front)
    avg_train_loss_top   = np.average(train_losses_top)
    avg_train_loss_align = np.average(train_losses_align)

    model.eval()
    val_losses_front, val_losses_top, val_losses_align = [], [], []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader),
                               desc=f"Epoch: {epoch+1}/{num_epochs} [VALID]",
                               total=len(val_loader)):

            images, kp_gt, kp_htmp_pred, kp_htmp_gt, reps = {}, {}, {}, {}, {}
            for view in [ "front", "top" ]:
                images[view] = batch["image"][view].to(device)
                kp_gt[view] = batch["keypoints"][view].to(device)
                kp_htmp_pred[view], reps[view] = model(images[view], ret_rep=True)
                if view == "top":
                    kp_htmp_pred[view] = kp_htmp_pred[view][:, :7, :, :]
                htmp_size = kp_htmp_pred[view].shape[2:]
                scale = kp_htmp_pred[view].shape[-1] / images[view].shape[-1]
                kp_htmp_gt[view] = generate_heatmaps(kp_gt[view], heatmap_size=htmp_size, sigma=sigma, scale=scale)
        
            loss_front: torch.Tensor = heatmap_loss_fn(kp_htmp_pred["front"], kp_htmp_gt["front"])
            loss_top:   torch.Tensor = heatmap_loss_fn(kp_htmp_pred["top"], kp_htmp_gt["top"])

            reps_front, reps_top = reps["front"][1], reps["top"][1]
            reps_front = [ torch.nn.functional.avg_pool3d(rep, 8) for rep in reps_front ]
            reps_top   = [ torch.nn.functional.avg_pool3d(rep, 8) for rep in reps_top ]
            # reps_front = [ torch.nn.functional.adaptive_avg_pool2d(rep, (rep.size(0), 512)) for rep in reps_front ]
            # reps_top   = [ torch.nn.functional.adaptive_avg_pool2d(rep, (rep.size(0), 512)) for rep in reps_top ]
            loss_align: torch.Tensor = contrastive_loss_fn(reps_front, reps_top)

            val_losses_front.append(loss_front.item())
            val_losses_top.append(loss_top.item())
            val_losses_align.append(loss_align.item())

            scale = 1 / scale
            if idx == 0:
                for i in range(images["front"].size(0)):
                    if i == 8: break
                    image = images["front"][i]
                    visualize_heatmaps(image, kp_htmp_pred["front"][i, :1], save_path=f"results/cl_hrnet/heatmap_{i}_front_pred.png")
                    visualize_heatmaps(image, kp_htmp_gt["front"][i, :1], save_path=f"results/cl_hrnet/heatmap_{i}_front_truth.png")
                    visualize_peaks_from_heatmap(image, kp_htmp_pred["front"][i], save_path=f"results/cl_hrnet/peaks_{i}_front_pred.png", scale=scale)
                    visualize_peaks_from_heatmap(image, kp_htmp_gt["front"][i], save_path=f"results/cl_hrnet/peaks_{i}_front_truth.png", scale=scale)

    avg_val_loss_front = np.average(val_losses_front)
    avg_val_loss_top   = np.average(val_losses_top)
    avg_val_loss_align = np.average(val_losses_align)

    print(f"Epoch [{epoch+1}/{num_epochs}]:\n" + \
          f"Training   Loss Front: {avg_train_loss_front:.8f}, Training   Loss Top: {avg_train_loss_top:.8f}, Training   Loss Align: {avg_train_loss_align:.8f}\n" + \
          f"Validation Loss Front: {avg_val_loss_front:.8f}, Validation Loss Top: {avg_val_loss_top:.8f}, Validation Loss Align: {avg_val_loss_align:.8f}\n" + \
          f"LR: {get_lr(optimizer):.10f}\n")
    
    with open("results/cl_hrnet/logs.txt", mode='a') as log_file:
        log_file.write(f"{epoch},")
        log_file.write(f"{avg_train_loss_front},{avg_train_loss_top},{avg_train_loss_align},")
        log_file.write(f"{avg_val_loss_front},{avg_val_loss_top},{avg_val_loss_align},")
        log_file.write(f"{get_lr(optimizer)}\n")
    