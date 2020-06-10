

import cv2
import os
import numpy as np
import argparse
#from matplotlib import pyplot as plt
#import plotly.express as px
#import plotly.graph_objects as go

from six.moves import urllib
import requests
from pdb import set_trace
import glob

import tensorflow as tf
import torch


# FILE_ID = "1QspxOJMDf_rAWVV7AU_Nc0rjo1_EPEDW"
# DESTINATION = '../face_mask_detection.zip'
SOURCE_URL = "https://cloud.tsinghua.edu.cn/d/af356cf803894d65b447/files/?p=%2FAIZOO%2F%E4%BA%BA%E8%84%B8%E5%8F%A3%E7%BD%A9%E6%A3%80%E6%B5%8B%E6%95%B0%E6%8D%AE%E9%9B%86.zip&dl=1"

def maybe_download(filename, work_directory):
	"""Download the data from website, unless it's already here."""
	if not tf.io.gfile.exists(work_directory):
		tf.io.gfile.makedirs(work_directory)
	filepath = os.path.join(work_directory, filename)
	
	if not tf.io.gfile.exists(filepath):
		print("start downloading dataset...")
		filepath, _ = urllib.request.urlretrieve(SOURCE_URL, filepath)
		# filepath = download_file_from_google_drive(FILE_ID, filepath)
	with tf.io.gfile.GFile(filepath) as f:
		print('Successfully downloaded', filename)
	return filepath


import zipfile
from pathlib import Path
def extract_images(filename, work_directory):
	
	dest = work_directory + "/FaceMaskDataset/"
	"""Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
	if not tf.io.gfile.exists(dest):
		print('Extracting', filename)
		tf.io.gfile.makedirs(dest)
	
		with zipfile.ZipFile(filename, 'r') as zip_ref:
			zip_ref.extractall(dest)

def download_extract(dest):
  filepath = maybe_download("FaceMaskDataset.zip", dest)
  extract_images(filepath, dest)

if __name__=="__main__":
	parser = argparse.ArgumentParser(description="FaceMaskDetection")
	
	parser.add_argument('--dest', type=str, default="../FaceMaskDataset", help='path to dataset.')
	args = parser.parse_args()
	download_extract(dest)