from torch.utils.data import Dataset
import torch
from transformers import SegformerImageProcessor
import numpy as np
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


class KvasirSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.feature_extractor = feature_extractor
        self.transforms = transforms
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        assert len(self.images) == len(self.masks)
        for img, mask in zip(self.images, self.masks):
            assert img == mask, f"Image {img} does not match mask {mask}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            return [self._get_single_item(i) for i in idx]
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_np = np.array(image)
        mask_np = np.array(mask)

        if self.transforms:
            augmented = self.transforms(image=image_np, mask=mask_np)
            image_np = augmented["image"]
            mask_np = augmented["mask"]
        else:
            image_np = torch.tensor(image_np).permute(2, 0, 1).float()
            mask_np = np.where(mask_np > 127, 1, 0)

        inputs = self.feature_extractor(images=[image_np], return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)

        mask = np.where(mask_np > 127, 1, 0)
        mask = torch.from_numpy(mask).long()

        return {"pixel_values": pixel_values, "labels": mask}


def kvasir_dataset(model_name, test_size, notebook=False):
    data_dir = "../data/Kvasir-SEG" if notebook else "data/Kvasir-SEG"
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    feature_extractor = SegformerImageProcessor.from_pretrained(model_name)

    transforms = A.Compose(
        [
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(),
        ]
    )

    dataset = KvasirSegDataset(image_dir, mask_dir, feature_extractor, transforms)

    total_size = len(dataset)
    test_size = int(total_size * test_size)
    train_size = total_size - test_size

    return torch.utils.data.random_split(dataset, [train_size, test_size])
