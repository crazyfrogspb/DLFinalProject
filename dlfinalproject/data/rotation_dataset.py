import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dlfinalproject.config import config


class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, random_state=24, train=True,
                 size_limits=(16, 64), num_regions=1):
        self.image_files = image_files
        self.size_limits = size_limits
        self.num_regions = num_regions
        self.train = train

        random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        np.random.seed(random_state)

        self.norm_tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=config.img_means, std=config.img_stds)])

    def __len__(self):
        if self.train:
            return len(self.image_files) * 4
        else:
            return len(self.image_files) 

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = int(idx.cpu().numpy())
        if self.train:
            image = Image.open(self.image_files[idx // 4])
        else:
            image = Image.open(self.image_files[idx])
        image = image.convert('RGB')
        #image = image.resize((224, 224), Image.LANCZOS)

        if not self.train:
            random.seed(idx)
            torch.manual_seed(idx)
            torch.cuda.manual_seed_all(idx)
            np.random.seed(idx)

        if self.train:
            rotation = idx % 4
        else:
            rotation = np.random.randint(0, 4)
        image = image.rotate(rotation * 90)

        image_tensor = self.norm_tf(image).to(config.device)

        return image_tensor, rotation
