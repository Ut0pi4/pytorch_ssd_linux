


import xml.etree.ElementTree as ET
import torch
import cv2
import os
import numpy as np

from six.moves import urllib
import requests
from pdb import set_trace
import glob

import tensorflow as tf
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    # set_trace()
    token = get_confirm_token(response)
    # set_trace()    
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    # set_trace()
    save_response_content(response, destination)    
    # set_trace()
    return destination

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    print_chunk = 100
    with open(destination, "wb") as f:
        for i, chunk in enumerate(response.iter_content(CHUNK_SIZE)):
            # print("chunk %d of %d done" %(i+1, CHUNK_SIZE))
            
            if ((i+1)%print_chunk==0):
                print('{:d} out of {:d} has been downloaded'.format(i, CHUNK_SIZE))
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

# FILE_ID = "1QspxOJMDf_rAWVV7AU_Nc0rjo1_EPEDW"
# DESTINATION = '../face_mask_detection.zip'
SOURCE_URL = "https://cloud.tsinghua.edu.cn/d/af356cf803894d65b447/files/?p=%2FAIZOO%2F%E4%BA%BA%E8%84%B8%E5%8F%A3%E7%BD%A9%E6%A3%80%E6%B5%8B%E6%95%B0%E6%8D%AE%E9%9B%86.zip&dl=1"

def maybe_download(filename, work_directory):
    """Download the data from website, unless it's already here."""
    if not tf.io.gfile.exists(work_directory):
        tf.io.gfile.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    # set_trace()
    if not tf.io.gfile.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL, filepath)
        # filepath = download_file_from_google_drive(FILE_ID, filepath)
    with tf.io.gfile.GFile(filepath) as f:
        print('Successfully downloaded', filename)
    return filepath


import zipfile
from pathlib import Path
def extract_images(filename, split):
    
    assert split in {'train', 'test'}
    
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    if not tf.io.gfile.exists("../FaceMaskDataset"):
        tf.io.gfile.makedirs("../FaceMaskDataset")
    
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall("../FaceMaskDataset")
    
    filepath = os.path.join(Path("../FaceMaskDataset/"+split), '*g')
    # filepath = os.path.join(Path("../FaskMaskDataset/train"), '*g')


    files = glob.glob(filepath)
    # set_trace()
    img_ids = []
    for img_id in files:
        img_ids.append(img_id)
    return img_ids


def load_data(data_dir = "./", batch_size = 1):
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])
    # }

    # image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms['train'])
    image_dataset = datasets.ImageFolder(os.path.join(data_dir)) 
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
    return data_loader

def loadxml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    size = []
    bnd_label = []
    bndbox = []
    for i in range(len(root)):
        if root[i].tag == 'size':
            for ele in root[i]:
                size.append(int(ele.text))
        elif root[i].tag == 'object':
            for ele in root[i]:
                if ele.tag == 'name':
                    bnd_label.append(ele.text)
                elif ele.tag == 'bndbox':
                    bnd = []
                    for ele_ in ele:
                        bnd.append(int(ele_.text))
                    bndbox.append(bnd)
    return [size, bnd_label, bndbox]

def resize_img(img, imgsize, min_size = 600, max_size = 1000):
    H, W = imgsize
    scale1 = min_size/min(H, W)
    scale2 = max_size/max(H, W)
    scale = min(scale1, scale2)
    img = cv2.resize(img.permute(1,2,0).numpy(), (int(W*scale), int(H*scale)))
    return img, scale

def resize_box(bbox, in_size, out_size):
    bbox = np.array(bbox).copy()
    y_scale = float(out_size[0]) / int(in_size[0])
    x_scale = float(out_size[1]) / int(in_size[1])
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox

def x_dataloader(dataloader, i):
    # set_trace()
    img = dataloader.dataset[i][0]
    path = dataloader.dataset.imgs[i][0]
    path = path[:-3]+'xml'
    imgsize, boxlabel, bndbox = loadxml(path)
    # set_trace()
    if (bndbox == []) | (imgsize == []) | (boxlabel == []) | (0 in imgsize):
        return [None, None, None]             
    # img, scale = resize_img(img, imgsize[:-1])
    # bndbox = resize_box(bndbox, imgsize[:-1], [scale*ele for ele in imgsize[:-1]])
    # boxlabel = boxlabel
    # set_trace()
    # return torch.from_numpy(img).permute(2,0,1).unsqueeze(0), \
    #             torch.from_numpy(bndbox), boxlabel
    return img, bndbox, boxlabel

def retrieve_gt(path, split, limit=0):

    filepath = maybe_download("FaceMaskDataset.zip", "../")
    image_ids = extract_images(filepath, split) 
    
    dataloader = load_data(data_dir = path)
    
    images = []
    bndboxes = []
    boxlabels = []
    # for i in range(len(dataloader)):
    N = len(dataloader)
    if limit:
      N = limit
    for i in range(N):
        img, bndbox, boxlabel = x_dataloader(dataloader, i)
        if img is None:
            continue
        boxlabel = [1 if label == "face" else 2 for label in boxlabel]
        images.append(img)
        bndboxes.append(bndbox)
        boxlabels.append(boxlabel)
        
    # boxlabel = ["face", "face_masks"]
    return images, bndboxes, boxlabels

if __name__=="__main__":
    retrieve_gt("../FaceMaskDataset", "train")
        