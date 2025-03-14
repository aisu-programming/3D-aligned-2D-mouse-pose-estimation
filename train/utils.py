import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt



def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def generate_heatmaps(keypoints, heatmap_size, sigma=2, scale=1):

    batch_size, num_joints, _ = keypoints.shape
    height, width = heatmap_size
    device = keypoints.device
    heatmaps = torch.zeros((batch_size, num_joints, height, width),
                           dtype=torch.float32, device=device)

    for b in range(batch_size):
        for j in range(num_joints):
            x, y, v = keypoints[b, j]
            x *= scale
            y *= scale
            if v > 0:
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(height, device=device),
                    torch.arange(width, device=device)
                )
                x_grid = x_grid.float()
                y_grid = y_grid.float()

                heatmap = torch.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
                heatmaps[b, j] = heatmap

    return heatmaps

def visualize_heatmaps(image: torch.Tensor, heatmaps: torch.Tensor, save_path=None):
    image: np.ndarray = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    heatmaps: np.ndarray = heatmaps.cpu().numpy()  # [num_joints, h, w]
    heatmap_max = np.max(heatmaps, axis=0)  # [h, w]
    
    heatmap_max = (heatmap_max - heatmap_max.min()) / (heatmap_max.max() - heatmap_max.min())
    heatmap_max = (heatmap_max * 255).astype(np.uint8)  # [h, w]

    heatmap_resized = cv2.resize(heatmap_max, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)  # [H, W]
    
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)  # [H, W, 3]
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    if save_path:
        plt.imsave(save_path, overlay)
    else:
        plt.imshow(overlay)
        plt.axis("off")
        plt.show()


def visualize_peaks_from_heatmap(image: torch.Tensor, heatmaps: torch.Tensor, threshold=0.1, save_path=None, scale=1):
    image: np.ndarray = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    
    heatmaps: np.ndarray = heatmaps.cpu().numpy()
    num_joints, _, _ = heatmaps.shape
    
    _, ax = plt.subplots(1)
    ax.imshow(image, cmap="gray")
    
    for i in range(num_joints):
        heatmap: np.ndarray = heatmaps[i]
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y *= scale
        x *= scale
        # max_value = heatmap[y, x]
        # if max_value > threshold:
        ax.plot(x, y, "ro", markersize=3)
        ax.text(x + 2, y - 2, f"{i}", color="yellow", fontsize=6)
    
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.show()


def visualize_peaks(image: torch.Tensor, peaks: torch.Tensor, threshold=0.1, save_path=None):
    image: np.ndarray = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    
    peaks: np.ndarray = peaks.cpu().numpy()
    
    _, ax = plt.subplots(1)
    ax.imshow(image)
    
    for i in range(peaks.shape[0]):
        x, y, _ = peaks[i]
        ax.plot(x, y, "ro", markersize=3)
        ax.text(x + 2, y - 2, f"{i}", color="yellow", fontsize=6)
    
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.show()


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.5, device="cpu"):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, z1_list: list[torch.Tensor], z2_list: list[torch.Tensor]):
        assert len(z1_list) == len(z2_list)

        loss = None
        for z1, z2 in zip(z1_list, z2_list):
            batch_size = z1.size(0)

            z1 = z1.view(batch_size, -1)  # (batch_size, c * w * h)
            z2 = z2.view(batch_size, -1)  # (batch_size, c * w * h)

            z1 = torch.nn.functional.normalize(z1, dim=1)  # (batch_size, c * w * h)
            z2 = torch.nn.functional.normalize(z2, dim=1)  # (batch_size, c * w * h)
            similarity_matrix = torch.matmul(z1, z2.T)  # (batch_size, batch_size)

            labels = torch.arange(batch_size).to(self.device)  # (batch_size,)
            logits = similarity_matrix / self.temperature
            if loss is None:
                loss = self.criterion(logits, labels)
            else:
                loss += self.criterion(logits, labels)

        return loss


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, device="cpu"):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, z1_list: list[torch.Tensor], z2_list: list[torch.Tensor]):
        assert len(z1_list) == len(z2_list)

        loss = None
        for z1, z2 in zip(z1_list, z2_list):
            batch_size = z1.size(0)

            # Flatten and normalize embeddings
            z1 = z1.view(batch_size, -1)  # (batch_size, c * w * h)
            z2 = z2.view(batch_size, -1)  # (batch_size, c * w * h)
            z1 = torch.nn.functional.normalize(z1, dim=1)  # Normalize (batch_size, c * w * h)
            z2 = torch.nn.functional.normalize(z2, dim=1)  # Normalize (batch_size, c * w * h)

            # Concatenate z1 and z2 for similarity computation
            z_cat = torch.cat([z1, z2], dim=0)  # (2 * batch_size, c * w * h)

            # Compute similarity matrix
            similarity_matrix = torch.matmul(z_cat, z_cat.T)  # (2 * batch_size, 2 * batch_size)
            similarity_matrix = similarity_matrix / self.temperature

            # Mask for positive pairs (diagonal offsets)
            labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)]).to(self.device)  # (2 * batch_size)

            # Exclude self-similarity
            mask = torch.eye(2 * batch_size, device=self.device).bool()
            similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

            # Compute the loss
            if loss is None:
                loss = self.criterion(similarity_matrix, labels)
            else:
                loss += self.criterion(similarity_matrix, labels)

        return loss

def multi_view_consistency_loss(keypoints_front, keypoints_top):
    """Loss to align keypoints between views."""
    return F.mse_loss(keypoints_front, keypoints_top)