import glob
import os.path as osp
from collections import OrderedDict

from tqdm import tqdm

import mlflow
import torch
from dlfinalproject.config import config
from dlfinalproject.data.rotation_dataset import RotationDataset
from dlfinalproject.submission import modelv2, revnet
from dlfinalproject.submission.model import Bottleneck, ResNet
from sklearn.model_selection import train_test_split


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


def train_model(image_folders, batch_size, test_size, random_state, early_stopping,
                learning_rate, decay, n_epochs, eval_interval,
                model_file, checkpoint_file, restart_optimizer, run_uuid,
                architecture, filters_factor, optim, momentum, patience):
    args_dict = locals()
    image_types = ['*.JPEG']
    image_files = []
    for image_folder in image_folders:
        for image_type in image_types:
            image_files.extend(
                glob.glob(osp.join(config.data_dir, 'raw', image_folder, '**', image_type), recursive=True))
    image_files = sorted(image_files)
    train_files, val_files = train_test_split(
        image_files, test_size=test_size, shuffle=True, random_state=random_state)
    dataset_train = RotationDataset(train_files)
    dataset_val = RotationDataset(val_files, train=False)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False)

    if architecture == 'resnet50':
        resnet = ResNet(Bottleneck, [3, 4, 6, 3],
                        filters_factor=filters_factor, num_classes=4)
    elif architecture == 'resnet152':
        resnet = ResNet(Bottleneck, [3, 8, 36, 3],
                        filters_factor=filters_factor, num_classes=4)
    elif architecture == 'resnet50v2':
        resnet = modelv2.ResNet(
            Bottleneck, [3, 4, 6, 3], filters_factor=filters_factor, num_classes=4)
    elif architecture == 'revnet50':
        resnet = revnet.RevNet(revnet.BottleneckRev, [
                               3, 4, 6, 3], filters_factor=filters_factor, num_classes=4)
    else:
        raise NotImplementedError(
            f'Architecture {architecture} is not implemented')

    resnet.train()
    resnet.to(config.device)

    if torch.cuda.device_count() > 1:
        resnet = torch.nn.DataParallel(resnet)

    if checkpoint_file:
        checkpoint = torch.load(osp.join(config.model_dir, checkpoint_file))
        try:
            resnet.load_state_dict(checkpoint['model'])
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                name = k[7:]
                new_state_dict[name] = v
            resnet.load_state_dict(new_state_dict)
    else:
        checkpoint = None

    criterion = torch.nn.CrossEntropyLoss()

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
        best_acc = checkpoint['best_acc']

    with mlflow.start_run(run_uuid=run_uuid):
        for key, value in args_dict.items():
            mlflow.log_param(key, value)
        for epoch_num in range(start_epoch, n_epochs):
            print('Epoch: ', epoch_num)
            for i, (imgs, labels) in enumerate(tqdm(loader_train, desc='training')):
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
                        for i, (imgs, labels) in enumerate(tqdm(loader_val, desc='validation')):
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
                                      'total_iterations': total_iterations}
                        torch.save(checkpoint, osp.join(
                            config.model_dir, model_file))
                    else:
                        early_counter += 1
                        if early_counter >= early_stopping:
                            print('Early stopping')
                            break
                    resnet.train()
