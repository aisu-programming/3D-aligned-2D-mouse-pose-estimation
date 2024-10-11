import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset



class Resize:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size  # (height, width)

    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        keypoints_rescaled = keypoints.copy()
        keypoints_rescaled[:, 0] = keypoints[:, 0] * (new_w / w)  # x 坐标
        keypoints_rescaled[:, 1] = keypoints[:, 1] * (new_h / h)  # y 坐标

        return {"image": image_resized, "keypoints": keypoints_rescaled}

class Pad:
    def __init__(self, output_size):
        self.output_size = output_size  # (height, width)

    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        pad_h = new_h - h
        pad_w = new_w - w

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=0
        )
        image = np.expand_dims(image, axis=-1)

        keypoints[:, 0] += left
        keypoints[:, 1] += top

        return {"image": image, "keypoints": keypoints}

class RandomHorizontalFlip:
    def __init__(self, p=0.5, img_width=640):
        self.p = p
        self.img_width = img_width

    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        if np.random.rand() < self.p:
            image = np.fliplr(image).copy()
            keypoints[:, 0] = self.img_width - keypoints[:, 0]
        return {"image": image, "keypoints": keypoints}

class ToTensor:
    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        image: np.ndarray = image.transpose((2, 0, 1))  # HWC to CHW
        image = torch.from_numpy(image).float() / 255.0
        keypoints = torch.from_numpy(keypoints).float()
        return {"image": image, "keypoints": keypoints}

class CocoKeypointsDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_ids = list(self.coco.imgs.keys())[:1500]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = f"{self.img_dir}/{img_info['file_name']}"
        image: np.ndarray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        keypoints_list = []
        for ann in anns:
            if "keypoints" in ann and ann["num_keypoints"] > 0:
                kp = np.array(ann["keypoints"]).reshape(-1, 3)  # (num_keypoints, 3)
                keypoints_list.append(kp)
        if keypoints_list:
            keypoints = np.concatenate(keypoints_list, axis=0)  # (N, 3)
        else:
            raise Exception

        sample = {"image": image, "keypoints": keypoints}

        if self.transform:
            sample = self.transform(sample)

        return sample
