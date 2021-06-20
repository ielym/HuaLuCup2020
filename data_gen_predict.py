# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import math
import codecs
import random
import numpy as np
from glob import glob
import cv2
from PIL import Image

from utils import scale_img_maxsize, AddGauseNoise, AddPepperNoise

class BaseDataset(Dataset):
	def __init__(self, img_files, h_w, max_size):
		self.samples = img_files
		self.h_w = h_w
		self.max_size = max_size

	def __len__(self):
		return len(self.samples)

	def preprocess_img1(self, img_path):
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		img = img.convert("RGB")
		ori_w, ori_h = img.size

		target_h1, target_w1, _, _ = scale_img_maxsize(ori_h, ori_w, maxsize=self.max_size[0])
		target_h2, target_w2, _, _ = scale_img_maxsize(ori_h, ori_w, maxsize=self.max_size[1])
		target_h3, target_w3, _, _ = scale_img_maxsize(ori_h, ori_w, maxsize=self.max_size[2])

		target_h1 = target_h1 if target_h1 % 2 == 0 else target_h1 + 1
		target_h2 = target_h2 if target_h2 % 2 == 0 else target_h2 + 1
		target_h3 = target_h3 if target_h3 % 2 == 0 else target_h3 + 1
		target_w1 = target_w1 if target_w1 % 2 == 0 else target_w1 + 1
		target_w2 = target_w2 if target_w2 % 2 == 0 else target_w2 + 1
		target_w3 = target_w3 if target_w3 % 2 == 0 else target_w3 + 1

		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

		preprocess1 = transforms.Compose([
			transforms.Resize((target_h1, target_w1), interpolation=Image.ANTIALIAS),
			transforms.Pad((int((self.max_size[0] - target_w1) // 2), int((self.max_size[0] - target_h1) // 2)), fill=0, padding_mode='constant'),
			transforms.ToTensor(),
			normalize,
		])
		preprocess2 = transforms.Compose([
			transforms.Resize((target_h2, target_w2), interpolation=Image.ANTIALIAS),
			transforms.Pad((int((self.max_size[1] - target_w2) // 2), int((self.max_size[1] - target_h2) // 2)), fill=0, padding_mode='constant'),
			transforms.ToTensor(),
			normalize,
		])
		preprocess3 = transforms.Compose([
			transforms.Resize((target_h3, target_w3), interpolation=Image.ANTIALIAS),
			transforms.Pad((int((self.max_size[2] - target_w3) // 2), int((self.max_size[2] - target_h3) // 2)), fill=0, padding_mode='constant'),
			transforms.ToTensor(),
			normalize,
		])
		img1 = preprocess1(img)
		img2 = preprocess2(img)
		img3 = preprocess3(img)
		return img1, img2, img3

	def __getitem__(self, idx):
		img_name = self.samples[idx, 0]
		x1, x2, x3 = self.preprocess_img1(img_name)
		return x1, x2, x3

def data_flow(base_dir, h_w, max_size):

	img_files = glob(os.path.join(base_dir, '*.*'))
	print('Test samples : {}'.format(len(img_files)))

	img_files = np.array(img_files).reshape([-1, 1])
	test_dataset = BaseDataset(img_files, h_w, max_size)

	return test_dataset, img_files