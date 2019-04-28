import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from dlfinalproject.config import config


class InpaintingDataset(torch.utils.data.Dataset):
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
        self.mask_tf = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = int(idx.cpu().numpy())
        base_image = Image.open(self.image_files[idx])
        base_image = base_image.convert('RGB')

        if not self.train:
            random.seed(idx)
            torch.manual_seed(idx)
            torch.cuda.manual_seed_all(idx)
            np.random.seed(idx)

        masks = []
        noise_image = base_image.copy()
        noise_image.convert('RGBA')
        for i in range(self.num_regions):
            size_x = np.random.randint(
                self.size_limits[0], self.size_limits[1])
            size_y = np.random.randint(
                self.size_limits[0], self.size_limits[1])
            position_x = np.random.randint(0, config.img_size[0] - size_x)
            position_y = np.random.randint(0, config.img_size[1] - size_y)

            mask = Image.new('RGBA', noise_image.size, (0, 0, 0, 0))
            draw_img = ImageDraw.Draw(noise_image)
            draw_mask = ImageDraw.Draw(mask)
            draw_img.rectangle(((position_x, position_y),
                                (position_x + size_x, position_y + size_y)), fill='white')
            draw_mask.rectangle(
                ((position_x, position_y), (position_x + size_x, position_y + size_y)), fill='white')

            masks.append(mask)

        mask = Image.new('RGBA', masks[0].size, (0, 0, 0, 0))
        noise_image = noise_image.convert('RGB')
        for m in masks:
            mask = Image.blend(mask, m, 1.0 / self.num_regions)

        mask = mask.convert('L')
        mask_np = np.array(mask)
        mask_np = np.expand_dims(np.where(mask_np != 0.0, 1.0, 0.0), 2)

        image_tensor = self.mask_tf(noise_image).to(config.device)
        gt_tensor = self.mask_tf(base_image).to(config.device)
        mask_tensor = self.mask_tf(mask_np).to(config.device).float()

        return image_tensor * 255, mask_tensor, gt_tensor * 255
