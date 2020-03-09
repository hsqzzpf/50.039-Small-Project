import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from PIL import Image

import xml.etree.ElementTree as ET


class CustomDataset(Dataset):

	def __init__(self, dir_csv, dir_img, transforms=None):
		"""
			dir_csv: should be like (img name, label)
			dir_img: image file holder
		"""
		self.dir_csv = dir_csv
		self.dir_img = dir_img
		self.transforms = transforms

		# needs get a set of tuple (img_name, label)
		self.data_frame = pd.read_csv(dir_csv)

	def __len__(self):
		return len(self.data_frame)

	def __getitem__(self, idx):
		img_name = os.path.join(self.dir_img,
                                self.data_frame.iloc[idx, 0])
		img = Image.open(img_name)
		if not self.transforms:
			img = self.transforms(img)
		
		label = self.data_frame.iloc[idx, 1]
		return (img, label)


