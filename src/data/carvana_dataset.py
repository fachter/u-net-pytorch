import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, limit_files=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        if limit_files is not None:
            self.images = self.images[:limit_files]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = np.array(Image.open(image_path).convert('RGB'))

        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_mask.gif'))
            mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
            mask[mask == 255.0] = 1.0
        else:
            mask = np.zeros_like(image)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask
