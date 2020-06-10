
from eval import evaluate
import tensorflow as tf
from pdb import set_trace
from dataload import retrieve_gt
import os
from utils import *
from datasets import FaceMaskDataset
from train import train
from model import SSD300, MultiBoxLoss
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

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
        self.epochs = 60

if __name__ == '__main__':
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
  
    images, bnd_boxes, labels, difficults = retrieve_gt("../FaceMaskDataset", "train")
    print("%d images has been retrieved" %len(images))
    # set_trace()

    
    print("finish loading images")

    train_dataset = FaceMaskDataset(images, bnd_boxes, labels, "train")
    
    train(config, train_dataset, model, optimizer, start_epoch)
    
    # Testing Phase
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                     
    checkpoint = "../checkpoint_ssd300.pth.tar"
    
    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    
    #train(config, train_dataset)
    print("loading images")

    images, bnd_boxes, labels, difficults = retrieve_gt("../FaceMaskDataset", "val")
    print("%d images has been retrieved" %len(images))
    # set_trace()
    print("finish loading images")
    test_dataset = FaceMaskDataset(images, bnd_boxes, labels, "test")
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                               collate_fn=test_dataset.collate_fn, num_workers=config.workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    
    
    
    # Switch to eval mode
    model.eval()
    evaluate(test_loader, model)
