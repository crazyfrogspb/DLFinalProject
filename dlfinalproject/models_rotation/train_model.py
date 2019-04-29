import glob
import os.path as osp

import mlflow
import torch
import torchvision.models as models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dlfinalproject.config import config
from dlfinalproject.data.rotation_dataset import RotationDataset


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
                model_file, checkpoint_file, restart_optimizer, run_uuid):
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

    resnet = models.resnet152(pretrained=False)
    resnet.fc = torch.nn.Linear(2048, 4)
    resnet.train()
    resnet.to(config.device)

    if torch.cuda.device_count() > 1:
        resnet = torch.nn.DataParallel(resnet)

    if checkpoint_file:
        checkpoint = torch.load(osp.join(config.model_dir, checkpoint_file))
        try:
            resnet.load_state_dict(checkpoint['model'])
        except Exception as e:
            for key, value in checkpoint['model'].items():
                key = key.replace('module.', '')
                try:
                    multi_getattr(resnet, f'{key}.data').copy_(value)
                except AttributeError:
                    print(f'Parameter {key} not found')
                except RuntimeError as e:
                    print(e)
    else:
        checkpoint = None

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        resnet.parameters(), lr=learning_rate, weight_decay=decay)
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
