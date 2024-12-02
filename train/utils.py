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
        ax.text(x + 2, y - 2, f"{i}", color="yellow", fontsize=4)
    
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
        ax.text(x + 2, y - 2, f"{i}", color="yellow", fontsize=4)
    
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.show()
