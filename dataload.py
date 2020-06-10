


import xml.etree.ElementTree as ET
import torch
import cv2
import os
import numpy as np

from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from six.moves import urllib
import requests
from pdb import set_trace
import glob

import tensorflow as tf
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
def extract_images(filename):
	
	
	"""Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
	if not tf.io.gfile.exists("../FaceMaskDataset"):
		print('Extracting', filename)
		tf.io.gfile.makedirs("../FaceMaskDataset")
	
		with zipfile.ZipFile(filename, 'r') as zip_ref:
			zip_ref.extractall("../FaceMaskDataset")
	
	

def load_data(data_dir = "./", batch_size = 1):
	# data_transforms = {
	#     'train': transforms.Compose([
	#         transforms.ToTensor(),
	#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	#     ])
	# }

	# image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms['train'])
	files = glob.glob(data_dir)
	
	image_dataset = datasets.ImageFolder(os.path.join(data_dir)) 
	data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
	return data_loader

def loadxml(path):
	tree = ET.parse(path)
	root = tree.getroot()
	size = []
	bnd_label = []
	bndbox = []
	difficult = []
	for i in range(len(root)):
		if root[i].tag == 'size':
			for ele in root[i]:
				size.append(int(ele.text))
		elif root[i].tag == 'object':
			for ele in root[i]:
				if ele.tag == 'name':
					bnd_label.append(ele.text)
				elif ele.tag == 'difficult':
					difficult.append((int)(ele.text))    
				elif ele.tag == 'bndbox':
					bnd = []
					for ele_ in ele:
						bnd.append(int(ele_.text))
					bndbox.append(bnd)
	return [size, bnd_label, bndbox, difficult]

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

def data_loader(dataloader, i):
	# set_trace()
	img = dataloader.dataset[i][0]
	path = dataloader.dataset.imgs[i][0]
	
	path = path[:-3]+'xml'
	imgsize, boxlabel, bndbox, difficult = loadxml(path)
	# set_trace()
	if (bndbox == []) | (imgsize == []) | (boxlabel == []) | (0 in imgsize):
		return [None, None, None, None]             
	# img, scale = resize_img(img, imgsize[:-1])
	# bndbox = resize_box(bndbox, imgsize[:-1], [scale*ele for ele in imgsize[:-1]])
	# boxlabel = boxlabel
	# set_trace()
	# return torch.from_numpy(img).permute(2,0,1).unsqueeze(0), \
	#             torch.from_numpy(bndbox), boxlabel
	return img, bndbox, boxlabel, difficult

def retrieve_gt(path, split, limit=0):

	assert split in ["train", "val"]
	filepath = maybe_download("FaceMaskDataset.zip", "../")
	extract_images(filepath) 


	dataloader = load_data(data_dir = path)

	images = []
	bndboxes = []
	boxlabels = []
	difficults = []
	# set_trace()
	if split == "train":
		i = 0
		N = 6120
		if limit:
			N= limit
	elif split == "val":
		i = 6120
		N = len(dataloader)

		if limit:
			N = i + limit

	while i < N:
		img, bndbox, boxlabel, difficult = data_loader(dataloader, i)
		i += 1
		if img is None:
			continue
		boxlabel = [1 if label == "face" else 2 for label in boxlabel]
		images.append(img)
		bndboxes.append(bndbox)
		boxlabels.append(boxlabel)
		difficults.append(difficult)

		
	print("finish retrieving data")

	# boxlabel = ["face", "face_masks"]
	return images, bndboxes, boxlabels, difficults

if __name__=="__main__":
	

	for split in ["train", "val"]:
		images, bndboxes, boxlabels, difficults = retrieve_gt("../FaceMaskDataset/", split)
		print("count labels and difficults...")
		counts_labels = np.zeros((2))
		for labels in boxlabels:
			for i in range(2):
				counts_labels[i] += np.sum(np.array(labels)==i+1)
			 
		counts_difficults = np.zeros((2))
		
		for difficult in difficults:
			for i in range(2):
				counts_difficults[i] += np.sum(np.array(difficult)==i)
		print(split)
		print("label counts: ", counts_labels)
		print("difficult counts: ", counts_difficults)
		
		counts = counts_labels.astype(int)
		title = "Counts of Masks labels (" + split +")"
		fig = go.Figure(
			
			data=go.Bar(
				orientation='h',
				x=counts,
				y=["no_mask", "mask"],
				
				text=counts,
				textposition='auto'
				),
			
			layout=go.Layout(
				title=title,
				showlegend=False,
				xaxis=go.layout.XAxis(showticklabels=False),
				yaxis=go.layout.YAxis(autorange='reversed'),
				width=750, height=400
			)
		)
		fig.show()
		
		counts = counts_difficults.astype(int)
		title = "Counts of difficult images (" + split +")"
		fig = go.Figure(
			
			data=go.Bar(
				orientation='h',
				x=counts,
				y=["not difficult", "difficult"],
				
				text=counts,
				textposition='auto'
				),
			
			layout=go.Layout(
				title=title,
				showlegend=False,
				xaxis=go.layout.XAxis(showticklabels=False),
				yaxis=go.layout.YAxis(autorange='reversed'),
				width=750, height=400
			)
		)
		fig.show()
	
	