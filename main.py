import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import FaceMaskDataset
from utils import *

from eval import evaluate
import tensorflow as tf
from pdb import set_trace
from dataload import retrieve_gt
import os

cudnn.benchmark = True

class Config():

    def __init__(self):
        # Data parameters
        self.data_folder = '../FaceMaskDataset'  # folder with data files
        self.keep_difficult = True  # use objects considered difficult to detect?
        
        # Model parameters
        # Not too many here since the SSD300 has a very specific structure
        self.n_classes = len(label_map)  # number of different types of objects
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Learning parameters
        path = "../checkpoint_ssd300.pth.tar"
        if os.path.exists(path):
          self.checkpoint = path  # path to model checkpoint, None if none
        else:
          self.checkpoint = None
        self.batch_size = 8  # batch size
        # self.iterations = 120 # number of iterations to train
        self.iterations = 120000 # number of iterations to train
        self.workers = 0  # number of workers for loading data in the DataLoader
        self.print_freq = 50  # print training status every __ batches
        self.lr = 1e-3  # learning rate
        self.decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
        self.decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
        self.momentum = 0.9  # momentum
        self.weight_decay = 5e-4  # weight decay
        self.grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
        self.epochs = 100


def train(config, train_dataset):
    """
    Training.
    """
    # global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    # set_trace()
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

    # Move to default device
    model = model.to(config.device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(config.device)
    
    # set_trace()
    # Custom dataloaders
                                     
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=config.workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    # epochs = config.iterations // (len(train_dataset) // 32)
    epochs = config.epochs
    # set_trace()
    decay_lr_at = [it // (len(train_dataset) // 32) for it in config.decay_lr_at]

    # Epochs
    print("start training....")
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, config.decay_lr_to)

        # One epoch's training
        train_one_epoch(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)
        
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train_one_epoch(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    config = Config()
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

        # Clip gradients, if necessary
        if config.grad_clip is not None:
            clip_gradient(optimizer, config.grad_clip)

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
    config = Config()
    
    #print("loading images")
    # set_trace()
    #images, bnd_boxes, labels = retrieve_gt("../FaceMaskDataset", "train")
    # set_trace()
    #print("finish loading images")
    #train_dataset = FaceMaskDataset(images, bnd_boxes, labels, "train")
    
    #train(config, train_dataset)
    print("loading images")
    images, bnd_boxes, labels = retrieve_gt("../FaceMaskDataset", "test", limit=10)
    # set_trace()
    print("finish loading images")
    test_dataset = FaceMaskDataset(images, bnd_boxes, labels, "test")
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                               collate_fn=test_dataset.collate_fn, num_workers=config.workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                     
    checkpoint = config.checkpoint
    
    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    
    # Switch to eval mode
    model.eval()
    evaluate(test_loader, model)
