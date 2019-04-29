import glob
import os.path as osp

import torch
import torchvision.models as models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dlfinalproject.config import config
from dlfinalproject.data.rotation_dataset import RotationDataset


def train_model(image_folders, batch_size, test_size, random_state, early_stopping,
                learning_rate, decay, n_epochs, eval_interval,
                model_file, checkpoint_file, restart_optimizer):
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

    inception = models.inception_v3(pretrained=False)
    inception.AuxLogits.fc = torch.nn.Linear(768, 4)
    inception.fc = torch.nn.Linear(2048, 4)
    inception.train()
    inception.to(config.device)

    if torch.cuda.device_count() > 1:
        inception = torch.nn.DataParallel(inception)

    if checkpoint_file:
        checkpoint = torch.load(osp.join(config.model_dir, checkpoint_file))
        inception.load_state_dict(checkpoint['model'])
    else:
        checkpoint = None

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        inception.parameters(), lr=learning_rate, weight_decay=decay)
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

    for epoch_num in range(start_epoch, n_epochs):
        print('Epoch: ', epoch_num)
        for i, (imgs, labels) in enumerate(tqdm(loader_train, desc='training')):
            optimizer.zero_grad()
            imgs = imgs.to(config.device)
            labels = labels.to(config.device)
            outputs, aux_outputs = inception(imgs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
            loss_train += loss.item()
            loss.backward()
            optimizer.step()

            current_iteration += 1
            total_iterations += 1

            if current_iteration % eval_interval == 0:
                loss_train /= (current_iteration * batch_size)
                print('Train loss: ', loss_train)
                current_iteration = 0
                loss_train = 0.0
                loss_val = 0.0
                correct = 0
                total = 0
                inception.eval()
                with torch.no_grad():
                    for i, (imgs, labels) in enumerate(tqdm(loader_val, desc='validation')):
                        imgs = imgs.to(config.device)
                        labels = labels.to(config.device)
                        outputs = inception(imgs)
                        loss = criterion(outputs, labels)
                        loss_val += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                loss_val /= total
                acc = correct / total
                print('Validation loss: ', loss_val)
                print('Accuracy: ', acc)
                if acc > best_acc:
                    early_counter = 0
                    best_acc = acc
                    checkpoint = {'model': inception.state_dict(),
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
                inception.train()
