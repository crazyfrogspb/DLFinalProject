import os.path as osp
from collections import OrderedDict

import cv2
import mlflow
import numpy as np
import torch
import torchvision.datasets as datasets
from PIL import Image
from tqdm import tqdm

from albumentations import (Blur, Compose, Cutout, GridDistortion,
                            HorizontalFlip, Normalize, OneOf,
                            OpticalDistortion, RandomBrightness,
                            RandomSizedCrop, RGBShift, ShiftScaleRotate,
                            VerticalFlip)
from albumentations.pytorch import ToTensor
from dlfinalproject.config import config
from dlfinalproject.submission import modelv2, revnet
from dlfinalproject.submission.model import Bottleneck, ResNet

AUG = {'light': {'p_fliph': 0.25, 'p_flipv': 0.1, 'p_aug': 0.1, 'p_crop': 0.1, 'p_cut': 0.1, 'p_ssr': 0.1},
       'medium': {'p_fliph': 0.5, 'p_flipv': 0.15, 'p_aug': 0.25, 'p_crop': 0.25, 'p_cut': 0.25, 'p_ssr': 0.25},
       'heavy': {'p_fliph': 0.5, 'p_flipv': 0.2, 'p_aug': 0.4, 'p_crop': 0.4, 'p_cut': 0.4, 'p_ssr': 0.4}}
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class AlbumentationsDataset(datasets.DatasetFolder):
    def __init__(self, root, transform=None,
                 target_transform=None, loader=default_loader,
                 samples_per_class=64, random_state=24):
        super().__init__(root, loader, IMG_EXTENSIONS,
                         transform=transform,
                         target_transform=target_transform)
        self.samples_per_class = samples_per_class
        targets = np.array(self.targets)
        indices = []
        np.random.seed(random_state)
        for class_num, class_name in enumerate(self.classes):
            class_inds = np.random.choice(np.argwhere(
                targets == class_num).reshape(-1), samples_per_class, replace=False)
            indices.extend(list(class_inds))
        self.indices = sorted(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        path, target = self.samples[self.indices[index]]
        sample = self.loader(path)
        if self.transform is not None:
            sample = np.array(sample)
            augmented = self.transform(image=sample)
            sample = augmented['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def multi_getattr(obj, attr, default=None):
    attributes = attr.split(".")
    for i in attributes:
        try:
            obj = getattr(obj, i)
        except AttributeError:
            if default:
                return default
            else:
                raise
    return obj


def image_loader(path, batch_size, augmentation=None,
                 samples_per_class=64, val_samples_per_class=64, random_state=24):
    if augmentation is None:
        transform = Compose([
            Normalize(mean=config.img_means, std=config.img_stds),
            ToTensor()
        ])
    else:
        transform = Compose([
            HorizontalFlip(p=AUG[augmentation]['p_fliph']),
            VerticalFlip(p=AUG[augmentation]['p_flipv']),
            OneOf([RandomSizedCrop((64, 90), 96, 96, p=1.0,
                                   interpolation=cv2.INTER_LANCZOS4)], p=AUG[augmentation]['p_crop']),
            ShiftScaleRotate(p=AUG[augmentation]['p_ssr'],
                             interpolation=cv2.INTER_LANCZOS4),
            OneOf([RGBShift(p=1.0),
                   RandomBrightness(p=1.0, limit=0.35),
                   Blur(p=1.0, blur_limit=4),
                   OpticalDistortion(p=1.0),
                   GridDistortion(p=1.0)], p=AUG[augmentation]['p_aug']),
            Cutout(num_holes=2, max_h_size=24, max_w_size=24,
                   p=AUG[augmentation]['p_cut']),
            Normalize(mean=config.img_means, std=config.img_stds),
            ToTensor()
        ])
    transform_val = Compose([
        Normalize(mean=config.img_means, std=config.img_stds),
        ToTensor()
    ])
    sup_train_data = AlbumentationsDataset(
        f'{path}/supervised/train', transform=transform, samples_per_class=samples_per_class, random_state=random_state)
    sup_val_data = AlbumentationsDataset(
        f'{path}/supervised/val', transform=transform_val, samples_per_class=val_samples_per_class, random_state=random_state)
    data_loader_sup_train = torch.utils.data.DataLoader(
        sup_train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    data_loader_sup_val = torch.utils.data.DataLoader(
        sup_val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    return data_loader_sup_train, data_loader_sup_val


def train_model(image_folders, batch_size, early_stopping,
                learning_rate, decay, n_epochs, eval_interval,
                model_file, checkpoint_file, restart_optimizer, run_uuid, finetune,
                augmentation, architecture, filters_factor, ignore_best_acc,
                optim, momentum, patience, samples_per_class, val_samples_per_class,
                random_state):
    args_dict = locals()
    data_loader_sup_train, data_loader_sup_val = image_loader(osp.join(
        config.data_dir, 'raw'), batch_size, augmentation, samples_per_class, val_samples_per_class, random_state)

    if architecture == 'resnet50':
        resnet = ResNet(Bottleneck, [3, 4, 6, 3], filters_factor=filters_factor)
    elif architecture == 'resnet152':
        resnet = ResNet(Bottleneck, [3, 8, 36, 3],
                        filters_factor=filters_factor)
    elif architecture == 'resnet50v2':
        resnet = modelv2.ResNet(
            Bottleneck, [3, 4, 6, 3], filters_factor=filters_factor)
    elif architecture == 'revnet50':
        resnet = revnet.RevNet(revnet.BottleneckRev, [
                               3, 4, 6, 3], filters_factor=filters_factor)
    resnet.train()
    resnet.to(config.device)

    if torch.cuda.device_count() > 1:
        resnet = torch.nn.DataParallel(resnet)

    if checkpoint_file:
        checkpoint = torch.load(osp.join(config.model_dir, checkpoint_file))
    else:
        checkpoint = None

    if checkpoint is not None:
        try:
            resnet.load_state_dict(checkpoint['model'])
        except RuntimeError:
            try:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['model'].items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                resnet.load_state_dict(new_state_dict)
            except Exception:
                for key, value in checkpoint['model'].items():
                    key = key.replace('module.', '')
                    try:
                        multi_getattr(resnet, f'{key}.data').copy_(value)
                    except AttributeError:
                        print(f'Parameter {key} not found')
                    except RuntimeError as e:
                        print(e)

    criterion = torch.nn.CrossEntropyLoss()

    if finetune == 'logistic':
        trainable_layers = ['fc']
    elif finetune == 'last':
        trainable_layers = ['layer4', 'avgpool', 'fc']
        if architecture in ['resnet50v2', 'revnet50']:
            trainable_layers.extend(['relu', 'bn_last'])
    elif finetune is None:
        trainable_layers = ['conv1', 'bn1', 'relu', 'maxpool',
                            'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
        if architecture in ['resnet50v2', 'revnet50']:
            trainable_layers.append('bn_last')
        if architecture == 'revnet50':
            trainable_layers.append('pool_double')

    for name, child in resnet.named_children():
        if name in trainable_layers:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False

    if optim == 'adam':
        optimizer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, resnet.parameters()), lr=learning_rate, weight_decay=decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, resnet.parameters(
        )), lr=learning_rate, weight_decay=decay, momentum=momentum)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, resnet.parameters(
        )), lr=learning_rate, weight_decay=decay, momentum=momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience, verbose=True)

    if checkpoint and not restart_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = 0
    total_iterations = 0
    current_iteration = 0
    loss_train = 0.0
    early_counter = 0
    best_acc = 0.0
    if checkpoint:
        start_epoch = checkpoint['epoch']
        total_iterations = checkpoint['total_iterations']
        run_uuid = checkpoint.get('run_uuid')
        if not ignore_best_acc:
            best_acc = checkpoint['best_acc']

    with mlflow.start_run(run_uuid=run_uuid):
        for key, value in args_dict.items():
            mlflow.log_param(key, value)
        for epoch_num in range(start_epoch, n_epochs):
            print('Epoch: ', epoch_num)
            for i, (imgs, labels) in enumerate(tqdm(data_loader_sup_train, desc='training')):
                optimizer.zero_grad()
                imgs = imgs.to(config.device)
                labels = labels.to(config.device)
                outputs = resnet(imgs)
                loss = criterion(outputs, labels)
                loss_train += loss.item()
                loss.backward()
                optimizer.step()

                current_iteration += 1
                total_iterations += 1

                if current_iteration % eval_interval == 0:
                    loss_train /= (current_iteration * batch_size)
                    print('Train loss: ', loss_train)
                    mlflow.log_metric('loss_train', loss_train)
                    current_iteration = 0
                    loss_train = 0.0
                    loss_val = 0.0
                    correct = 0
                    total = 0
                    resnet.eval()
                    with torch.no_grad():
                        for i, (imgs, labels) in enumerate(tqdm(data_loader_sup_val, desc='validation')):
                            imgs = imgs.to(config.device)
                            labels = labels.to(config.device)
                            outputs = resnet(imgs)
                            loss = criterion(outputs, labels)
                            loss_val += loss.item()
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    loss_val /= total
                    acc = correct / total
                    scheduler.step(acc)
                    print('Validation loss: ', loss_val)
                    mlflow.log_metric('loss_val', loss_val)
                    print('Accuracy: ', acc)
                    mlflow.log_metric('accuracy', acc)
                    if acc > best_acc:
                        early_counter = 0
                        best_acc = acc
                        checkpoint = {'model': resnet.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'epoch': epoch_num,
                                      'best_acc': best_acc,
                                      'total_iterations': total_iterations,
                                      'run_uuid': run_uuid}
                        torch.save(checkpoint, osp.join(
                            config.model_dir, model_file))
                    else:
                        early_counter += 1
                        if early_counter >= early_stopping:
                            print('Early stopping')
                            break
                    resnet.train()
