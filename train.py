import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import argparse
from model import SSD300, MultiBoxLoss
from config import Config
from datasets import FaceMaskDataset
from utils import *
from dataload import retrieve_gt
#from eval import evaluate
import tensorflow as tf
from pdb import set_trace
from dataload import retrieve_gt
import os

cudnn.benchmark = True

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def train(config, train_dataset, model, optimizer, start_epoch):
    """
    Training.
    """
    # global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    # set_trace()
    

    # Move to default device
    model = model.to(config.device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(config.device)
    
    # set_trace()
    # Custom dataloaders
                                     
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=config.workers,
                                               pin_memory=True)  # note that we're passing the collate function here


    epochs = config.epochs


    # Epochs
    print("start training....")
    for epoch in range(start_epoch, epochs):


        # One epoch's training
        train_one_epoch(config, train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)
        
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    
    # Batches
    # for i, (images, boxes, labels, _) in enumerate(train_loader):
    for i, (images, boxes, labels) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(config.device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(config.device) for b in boxes]
        labels = [l.to(config.device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()


        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="FaceMaskDetection")
    
    parser.add_argument('--dest', type=str, default="../FaceMaskDataset", help='path to dataset.')
    parser.add_argument('--limit', type=int, default=0, help='limit number of images.')
    
    args = parser.parse_args()
    config = Config()
    
    # Training Phase
    if config.checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=config.n_classes)
        # set_trace()
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * config.lr}, {'params': not_biases}],
                                    lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    
    else:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    
    print("loading images")
    
    images, bnd_boxes, labels, difficults = retrieve_gt(args.dest, "train", limit=args.limit)
    print("%d images has been retrieved" %len(images))
    # set_trace()
    
    
    print("finish loading images")
    
    train_dataset = FaceMaskDataset(images, bnd_boxes, labels, "train")
    
    train(config, train_dataset, model, optimizer, start_epoch)
