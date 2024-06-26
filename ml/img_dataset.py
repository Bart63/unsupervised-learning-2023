import os
import random
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from PIL import Image


class ImageHalfDataset(VisionDataset):
    def __init__(self, root, transform=None, first_half=True):
        super(ImageHalfDataset, self).__init__(root, transform=transform)
        self.first_half = first_half
        self.image_paths = sorted([os.path.join(root, img) for img in os.listdir(root)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        width, height = image.size

        if self.first_half:
            image = image.crop((0, 0, width // 2, height))
        else:
            image = image.crop((width // 2, 0, width, height))

        if self.transform:
            image = self.transform(image)

        return image


data_dir = '../dataset/1char_all_full/pairs_e22_k2_s0'

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to PyTorch tensor
])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_emnist(batch_size=64, num_workers=2, seed=0) -> DataLoader:
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(0)
    first_half_dataset = ImageHalfDataset(data_dir, transform=transform, first_half=True)
    first_half_dataloader = DataLoader(
        first_half_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=seed_worker, generator=g
    )
    return first_half_dataloader


def get_kmnist(batch_size=64, num_workers=2, seed=0) -> DataLoader:
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(0)
    second_half_dataset = ImageHalfDataset(data_dir, transform=transform, first_half=False)
    second_half_dataloader = DataLoader(
        second_half_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=seed_worker, generator=g
    )
    return second_half_dataloader
