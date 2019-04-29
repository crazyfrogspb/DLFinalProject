import os.path as osp

import cv2
import mlflow
import torch
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm

from albumentations import (Blur, CenterCrop, Compose, Flip, GridDistortion,
                            Normalize, OneOf, RandomBrightness, RandomCrop,
                            Resize, RGBShift, ShiftScaleRotate, ToFloat)
from dlfinalproject.config import config

AUG = {'disable': {'p_flip': 0.0, 'p_aug': 0.0}, 'light': {'p_flip': 0.25, 'p_aug': 0.1},
       'medium': {'p_flip': 0.5, 'p_aug': 0.25}, 'heavy': {'p_flip': 0.5, 'p_aug': 0.5}}


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


def image_loader(path, batch_size, augmentation=None):
    if augmentation is None:
        augmentation = 'disable'
    transform = Compose([
        Flip(p=AUG[augmentation]['p_flip']),
        OneOf([RandomCrop(200, 200, p=1.0),
               CenterCrop(200, 200, p=1.0),
               ShiftScaleRotate(p=1.0, interpolation=cv2.INTER_LANCZOS4),
               RGBShift(p=1.0),
               RandomBrightness(p=1.0),
               Blur(p=1.0),
               GridDistortion(p=1.0)], p=AUG[augmentation]['p_aug']),
        Resize(224, 224, interpolation=cv2.INTER_LANCZOS4),
        ToFloat(),
        Normalize(mean=config.img_means, std=config.img_stds)
    ])
    transform_val = Compose([
        Resize(224, 224, interpolation=cv2.INTER_LANCZOS4),
        ToFloat(),
        Normalize(mean=config.img_means, std=config.img_stds)
    ])
    sup_train_data = datasets.ImageFolder(
        f'{path}/supervised/train', transform=transform)
    sup_val_data = datasets.ImageFolder(
        f'{path}/supervised/val', transform=transform_val)
    data_loader_sup_train = torch.utils.data.DataLoader(
        sup_train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    data_loader_sup_val = torch.utils.data.DataLoader(
        sup_val_data, batch_size=batch_size, shuffle=True, num_workers=0)
    return data_loader_sup_train, data_loader_sup_val


def train_model(image_folders, batch_size, early_stopping,
                learning_rate, decay, n_epochs, eval_interval,
                model_file, checkpoint_file, restart_optimizer, run_uuid, finetune,
                augmentation):
    args_dict = locals()
    data_loader_sup_train, data_loader_sup_val = image_loader(
        osp.join(config.data_dir, 'raw'), batch_size, augmentation)

    resnet = models.resnet152(pretrained=False)
    resnet.train()
    resnet.to(config.device)

    if torch.cuda.device_count() > 1:
        resnet = torch.nn.DataParallel(resnet)

    if checkpoint_file:
        checkpoint = torch.load(osp.join(config.model_dir, checkpoint_file))
        resnet.load_state_dict(checkpoint['model'])
    else:
        checkpoint = None

    if checkpoint is not None:
        try:
            resnet.load_state_dict(checkpoint['model'])
        except Exception as e:
            for key, value in checkpoint['model'].items():
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
    elif finetune is None:
        trainable_layers = ['conv1', 'bn1', 'relu', 'maxpool',
                            'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']

    for name, child in resnet.named_children():
        if name in trainable_layers:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False

    optimizer = torch.optim.Adam(filter(
        lambda p: p.requires_grad, resnet.parameters()), lr=learning_rate, weight_decay=decay)
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
                                      'total_iterations': total_iterations}
                        torch.save(checkpoint, osp.join(
                            config.model_dir, model_file))
                    else:
                        early_counter += 1
                        if early_counter >= early_stopping:
                            print('Early stopping')
                            break
                    resnet.train()
