from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = torch.tensor([0.393692363414, 0.401524053519, 0.387328566137])[:, None, None]
STD = torch.tensor([0.242158539691, 0.236066870034, 0.244438255783])[:, None, None]

class XiamenBus(Dataset):
    def __init__(self, root, split='test', augment=False):
        self.root = Path(root)
        self.base = self.root / (split + '_data')
        self.names = (self.root / 'splits' / (split + '.txt')).read_text().splitlines()
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.base / 'images' / name) as im:
            image = np.array(im.convert('RGB'), dtype=np.float32) / 255.
        density = np.loadtxt(self.base / 'ground_truth' / (Path(name).stem + '.csv'), delimiter=',', dtype=np.float32)
        if density.shape != image.shape[:2]:
            raise ValueError(f'Image/density shape mismatch: {name}')
        if self.augment and random.random() < 0.5:
            image = image[:, ::-1].copy()
            density = density[:, ::-1].copy()
        x = (torch.from_numpy(image).permute(2, 0, 1) - MEAN) / STD
        y = torch.from_numpy(density.copy()).unsqueeze(0)
        return x, y, name
