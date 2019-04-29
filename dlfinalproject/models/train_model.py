import glob
import os.path as osp

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dlfinalproject.config import config
from dlfinalproject.data.inpainting_dataset import InpaintingDataset
from dlfinalproject.models.loss import GeneratorLoss, HingeLoss, L1Loss
from dlfinalproject.models.networks import Discriminator, Generator


def img2photo(img):
    return ((img + 1) / 2)


def evaluate(netG, netD, loader_val, recon_loss, gan_loss, dis_loss):
    netG.eval()
    netD.eval()
    loss_val = {'total': 0.0, 'gan': 0.0, 'rec': 0.0, 'dis': 0.0}

    counter = 0

    with torch.no_grad():
        for image, mask, gt in tqdm(loader_val, desc='validation'):
            image = image / 127.5 - 1
            gt = gt / 127.5 - 1
            coarse_imgs, recon_imgs = netG(image, mask)
            complete_imgs = recon_imgs * mask + image * (1 - mask)
            pos_imgs = torch.cat([gt, mask], dim=1)
            neg_imgs = torch.cat([complete_imgs, mask], dim=1)
            pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)
            pred_pos_neg = netD(pos_neg_imgs)
            pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
            d_loss = dis_loss(pred_pos, pred_neg)
            g_loss = gan_loss(pred_neg)
            r_loss = recon_loss(gt, recon_imgs, coarse_imgs)
            whole_loss = g_loss + r_loss
            loss_val['total'] += whole_loss.item()
            loss_val['gan'] += g_loss.item()
            loss_val['rec'] += r_loss.item()
            loss_val['dis'] += d_loss.item()
            counter += 1

    mask = mask.expand(mask.size(0), 3, mask.size(2), mask.size(3))
    image = img2photo(image)
    coarse_imgs = img2photo(coarse_imgs)
    recon_imgs = img2photo(recon_imgs)
    complete_imgs = img2photo(complete_imgs)
    gt = img2photo(gt)
    grid = make_grid(
        torch.cat((image, mask, coarse_imgs,
                   recon_imgs, complete_imgs,
                   gt), dim=0), nrow=image.size(0))
    for key, value in loss_val.items():
        loss_val[key] = value / counter

    return loss_val, grid


def train_model(image_folders, batch_size, test_size, random_state,
                learning_rate, decay, n_epochs, eval_interval,
                model_file, checkpoint_file, restart_optimizer, ignore_best_loss):
    image_types = ['*.JPEG']
    image_files = []
    for image_folder in image_folders:
        for image_type in image_types:
            image_files.extend(
                glob.glob(osp.join(config.data_dir, 'raw', image_folder, '**', image_type), recursive=True))
    image_files = sorted(image_files)
    train_files, val_files = train_test_split(
        image_files, test_size=test_size, shuffle=True, random_state=random_state)
    dataset_train = InpaintingDataset(train_files)
    dataset_val = InpaintingDataset(val_files, train=False)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False)

    netG = Generator(n_in_channel=5, batch_norm=True).to(config.device)
    netD = Discriminator(n_in_channel=4, batch_norm=True).to(config.device)
    netG.train()
    netD.train()

    if torch.cuda.device_count() > 1:
        netG = torch.nn.DataParallel(netG)
        netD = torch.nn.DataParallel(netD)

    if checkpoint_file:
        checkpoint = torch.load(osp.join(config.model_dir, checkpoint_file))
        netG.load_state_dict(checkpoint['netG'])
        netD.load_state_dict(checkpoint['netD'])
    else:
        checkpoint = None

    recon_loss = L1Loss()
    gan_loss = GeneratorLoss()
    dis_loss = HingeLoss()

    optG = torch.optim.Adam(
        netG.parameters(), lr=learning_rate, weight_decay=decay)
    optD = torch.optim.Adam(
        filter(lambda p: p.requires_grad, netD.parameters()), lr=4 * learning_rate, weight_decay=decay)
    if checkpoint and not restart_optimizer:
        optG.load_state_dict(checkpoint['optG'])
        optD.load_state_dict(checkpoint['optD'])

    start_epoch = 0
    total_iterations = 0
    current_iteration = 0
    loss_train = {'total': 0.0, 'gan': 0.0, 'rec': 0.0, 'dis': 0.0}
    LAMBDA = 1.0

    if checkpoint:
        start_epoch = checkpoint['epoch']
        total_iterations = checkpoint['total_iterations']

    for epoch_num in range(start_epoch, n_epochs):
        print('Epoch: ', epoch_num)
        for i, (image, mask, gt) in enumerate(tqdm(loader_train, desc='training')):
            optD.zero_grad(), netD.zero_grad(), netG.zero_grad(), optG.zero_grad()
            image = image / 127.5 - 1
            gt = gt / 127.5 - 1
            coarse_imgs, recon_imgs = netG(image, mask)
            complete_imgs = recon_imgs * mask + image * (1 - mask)
            pos_imgs = torch.cat([gt, mask], dim=1)
            neg_imgs = torch.cat([complete_imgs, mask], dim=1)
            pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)
            pred_pos_neg = netD(pos_neg_imgs)
            pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
            d_loss = dis_loss(pred_pos, pred_neg)
            d_loss.backward(retain_graph=True)
            total_norm_d = 0.0
            for p in netD.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_d += param_norm.item() ** 2
            total_norm_d = total_norm_d ** (1. / 2)
            optD.step()

            optD.zero_grad(), netD.zero_grad(), optG.zero_grad(), netG.zero_grad()
            g_loss = gan_loss(pred_neg)
            r_loss = recon_loss(gt, recon_imgs, coarse_imgs)
            whole_loss = g_loss + LAMBDA * r_loss
            whole_loss.backward()
            loss_train['total'] += whole_loss.item()
            loss_train['gan'] += g_loss.item()
            loss_train['rec'] += r_loss.item()
            loss_train['dis'] += d_loss.item()
            total_norm_g = 0.0
            for p in netG.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_g += param_norm.item() ** 2
            total_norm_g = total_norm_g ** (1. / 2)
            optG.step()

            current_iteration += 1
            total_iterations += 1

            if current_iteration % eval_interval == 0:
                print('Discriminator norm: ', total_norm_d)
                print('Generator norm: ', total_norm_g)
                current_iteration = 0
                loss_val, grid = evaluate(
                    netG, netD, loader_val, recon_loss, gan_loss, dis_loss)
                checkpoint = {'netG': netG.state_dict(),
                              'netD': netD.state_dict(),
                              'optG': optG.state_dict(),
                              'optD': optD.state_dict(),
                              'epoch': epoch_num,
                              'loss_val': loss_val,
                              'total_iterations': total_iterations}
                torch.save(checkpoint, osp.join(
                    config.model_dir, model_file))
                print('Validation loss')
                for key, value in loss_val.items():
                    print(f'{key} loss: {value}')
                save_image(grid, f'current_example.png')
                netG.train()
                netD.train()
