from pathlib import Path
from typing import Sequence

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image
