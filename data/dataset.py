import os
import cv2
import json
import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from deprecated import deprecated



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

        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        keypoints_rescaled = keypoints.copy()
        keypoints_rescaled[:, 0] = keypoints[:, 0] * (new_w / w)  # x 坐标
        keypoints_rescaled[:, 1] = keypoints[:, 1] * (new_h / h)  # y 坐标

        return {"image": image_resized, "keypoints": keypoints_rescaled}


class PairedResize:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size  # (height, width)

    def __call__(self, sample):
        output = { "image": {}, "keypoints": {} }
        for view in [ "front", "top" ]:
            image, keypoints = sample["image"][view], sample["keypoints"][view]
            h, w = image.shape[:2]
            new_h, new_w = self.output_size
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            keypoints_rescaled = keypoints.copy()
            keypoints_rescaled[:, 0] = keypoints[:, 0] * (new_w / w)  # x 坐标
            keypoints_rescaled[:, 1] = keypoints[:, 1] * (new_h / h)  # y 坐标
            output["image"][view] = image_resized
            output["keypoints"][view] = keypoints_rescaled
        return output


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

        keypoints[:, 0] += left
        keypoints[:, 1] += top

        return {"image": image, "keypoints": keypoints}


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        _, w = image.shape
        if np.random.rand() < self.p:
            image = np.fliplr(image).copy()
            keypoints[:, 0] = w - keypoints[:, 0]
        return {"image": image, "keypoints": keypoints}
    

class PairedRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p:
            output = { "image": {}, "keypoints": {} }
            for view in [ "front", "top" ]:
                image, keypoints = sample["image"][view], sample["keypoints"][view]
                _, w = image.shape
                image = np.fliplr(image).copy()
                keypoints[:, 0] = w - keypoints[:, 0]
                output["image"][view] = image
                output["keypoints"][view] = keypoints
            return output
        else:
            return sample
    

class RandomRotate90Degree:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        if np.random.rand() < self.p:
            image = np.rot90(image).copy()
            h, _ = image.shape
            keypoints[:, 0] = h - keypoints[:, 0]
            keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
        return {"image": image, "keypoints": keypoints}
    

class PairedRandomRotate90Degree:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p:
            output = { "image": {}, "keypoints": {} }
            for view in [ "front", "top" ]:
                image, keypoints = sample["image"][view], sample["keypoints"][view]
                image = np.rot90(image).copy()
                h, _ = image.shape
                keypoints[:, 0] = h - keypoints[:, 0]
                keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
                output["image"][view] = image
                output["keypoints"][view] = keypoints
            return output
        else:
            return sample


class ToTensor:
    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        image = np.expand_dims(image, axis=-1)
        image: np.ndarray = image.transpose((2, 0, 1))  # HWC to CHW
        image = torch.from_numpy(image).float() / 255.0
        keypoints = torch.from_numpy(keypoints).float()
        return {"image": image, "keypoints": keypoints}


class PairedToTensor:
    def __call__(self, sample):
        output = { "image": {}, "keypoints": {} }
        for view in [ "front", "top" ]:
            image, keypoints = sample["image"][view], sample["keypoints"][view]
            image = np.expand_dims(image, axis=-1)
            image: np.ndarray = image.transpose((2, 0, 1))  # HWC to CHW
            image = torch.from_numpy(image).float() / 255.0
            keypoints = torch.from_numpy(keypoints).float()
            output["image"][view] = image
            output["keypoints"][view] = keypoints
        return output


class KeypointsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        For handling keypoint data from the original dataset
        :param annotations_file: Path to the JSON file containing the data.
        :param img_dir: Directory where the images are stored.
        :param transform: Optional transformations or augmentations to apply to the data.
        """
        with open(annotations_file, "r") as f:
            self.data = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sample.
        """
        sample = self.data[idx]

        # Load the image
        img_path = os.path.join(self.img_dir, sample["filename"])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image {img_path} not found!")

        # Process keypoint data
        keypoints = []
        for color, coords in sample["coords"].items():
            x = np.array(coords["x"])
            y = np.array(coords["y"])
            # Mimic COCO keypoint format; assume all keypoints are visible (visibility=2)
            visibility = np.ones_like(x) * 2
            keypoints.append(np.stack([x, y, visibility], axis=-1))  # (num_keypoints, 3)
        
        # Combine keypoints from all colors
        keypoints = np.concatenate(keypoints, axis=0)  # (N, 3)

        # Package the data
        sample = {"image": image, "keypoints": keypoints}

        # Apply transformations (if any)
        if self.transform:
            sample = self.transform(sample)

        return sample
    

class CroppedKeypointsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        For handling keypoint data from the cropped JSON file
        :param annotations_file: Path to the JSON file containing the data.
        :param img_dir: Directory where the images are stored.
        :param transform: Optional transformations or augmentations to apply to the data.
        """
        with open(annotations_file, "r") as f:
            self.data = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sample.
        """
        sample = self.data[idx]

        # Load the image
        img_path = os.path.join(self.img_dir, sample["filename"])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image {img_path} not found!")

        # Process keypoint data
        x = np.array(sample["coords"]["x"])
        y = np.array(sample["coords"]["y"])
        # Mimic COCO keypoint format; assume all keypoints are visible (visibility=2)
        visibility = np.ones_like(x) * 2
        keypoints = np.stack([x, y, visibility], axis=-1)  # (num_keypoints, 3)

        # Package the data
        sample = {
            "image": image,
            "keypoints": keypoints,
            "class_name": sample["class_name"]  # Add class_name to the sample
        }

        # Apply transformations (if any)
        if self.transform:
            sample = self.transform(sample)

        return sample
    

class PairedCroppedKeypointsDataset(Dataset):
    def __init__(self, annotations_file_front, img_dir_front, annotations_file_top, img_dir_top, transform=None):
        """
        For handling keypoint data from the cropped JSON file
        :param annotations_file_front: Path to the JSON file containing the front data.
        :param img_dir_front: Directory where the front images are stored.
        :param annotations_file_top: Path to the JSON file containing the top data.
        :param img_dir_top: Directory where the top images are stored.
        :param transform: Optional transformations or augmentations to apply to the data.
        """
        with open(annotations_file_front, "r") as f:
            data_front = json.load(f)
        with open(annotations_file_top, "r") as f:
            data_top = json.load(f)

        front_dict = {
            f"{item['filename'].split('_')[2]}_{item['class_name']}": item
            for item in data_front
        }
        top_dict = {
            f"{item['filename'].split('_')[2]}_{item['class_name']}": item
            for item in data_top
        }

        combined_data = []
        for key, front_item in front_dict.items():
            if key in top_dict:
                top_item = top_dict[key]
                combined_data.append({
                    # "id": key,
                    "filename": {
                        "front": front_item["filename"],
                        "top": top_item["filename"],
                    },
                    "coords": {
                        "front": front_item["coords"],
                        "top": top_item["coords"],
                    },
                    # "class_name": {
                    #     "front": front_item["class_name"],
                    #     "top": top_item["class_name"],
                    # },
                })

        self.data = combined_data
        self.img_dir = {
            "front": img_dir_front,
            "top": img_dir_top,
        } 
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sample.
        """
        sample = self.data[idx]

        # Load the image
        img_path, image, keypoints = {}, {}, {}
        for view in [ "front", "top" ]:
            img_path[view] = os.path.join(self.img_dir[view], sample[f"filename"][view])
            image[view] = cv2.imread(img_path[view], cv2.IMREAD_GRAYSCALE)
            if image[view] is None:
                raise FileNotFoundError(f"Image {img_path[view]} not found!")

            # Process keypoint data
            x = np.array(sample["coords"][view]["x"])
            y = np.array(sample["coords"][view]["y"])
            # Mimic COCO keypoint format; assume all keypoints are visible (visibility=2)
            visibility = np.ones_like(x) * 2
            keypoints[view] = np.stack([x, y, visibility], axis=-1)  # (num_keypoints, 3)

        # Package the data
        sample = {
            "image": image,
            "keypoints": keypoints,
            # "class_name": sample["class_name"]  # Add class_name to the sample
        }

        # Apply transformations (if any)
        if self.transform:
            sample = self.transform(sample)

        return sample


@deprecated
class CocoKeypointsDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_ids = list(self.coco.imgs.keys())[:500]

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
