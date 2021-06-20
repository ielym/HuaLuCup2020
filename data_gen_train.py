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
from sklearn.model_selection import StratifiedShuffleSplit
from torchtoolbox.transform import Cutout
from PIL import Image

from utils import scale_img_maxsize, AddGauseNoise, AddPepperNoise

class BaseDataset(Dataset):
	def __init__(self, paths_labels, input_shape, mode):
		self.samples = paths_labels
		self.input_shape = input_shape
		self.mode = mode

	def __len__(self):
		return len(self.samples)

	def preprocess_img(self, img_path):
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		img = img.convert("RGB")
		ori_w, ori_h = img.size
		target_h, target_w, scale_h, scale_w = scale_img_maxsize(ori_h, ori_w, maxsize=self.input_shape[1])
		target_h = target_h if target_h % 2 == 0 else target_h + 1
		target_w = target_w if target_w % 2 == 0 else target_w + 1
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		preprocess = transforms.Compose([
			transforms.Resize((target_h, target_w), interpolation=Image.ANTIALIAS),
			AddGauseNoise(sigma=None, p=0.2),
			transforms.Pad((int((self.input_shape[1] - target_w) // 2), int((self.input_shape[1] - target_h) // 2)), fill=0, padding_mode='constant'),
			transforms.ToTensor(),
			normalize,
		])
		img = preprocess(img)
		return img

	def __getitem__(self, idx):

		x = self.preprocess_img(self.samples[idx, 0])
		y = np.array(self.samples[idx, 1:], dtype=np.float32)
		y = torch.tensor(y, dtype=torch.float32)
		return x, y


def data_flow(base_dir, input_shape):
	input_shape = random.choice([(3, 404, 404), (3, 436, 436), (3, 468, 468), (3, 500, 500)])

	train_dir = os.path.join(base_dir, 'train')
	train_txt = os.path.join(train_dir, 'train.txt')
	val_txt = os.path.join(train_dir, 'val_test.txt')

	train_calling_images = []
	train_normal_images = []
	train_smoking_images = []
	train_smoking_images_images = []
	with open(train_txt, 'r') as f:
		train_lines = f.readlines()
	for line in train_lines:
		line = line.strip()
		if line.startswith('normal'):
			train_normal_images.append(os.path.join(train_dir, line))
		elif line.startswith('smoking_calling'):
			train_smoking_images_images.append(os.path.join(train_dir, line))
		elif line.startswith('calling'):
			train_calling_images.append(os.path.join(train_dir, line))
		elif line.startswith('smoking'):
			train_smoking_images.append(os.path.join(train_dir, line))

	train_calling_images = np.array(train_calling_images).reshape([-1, 1])
	train_normal_images = np.array(train_normal_images).reshape([-1, 1])
	train_smoking_images = np.array(train_smoking_images).reshape([-1, 1])
	train_smoking_images_images = np.array(train_smoking_images_images).reshape([-1, 1])

	train_calling_labels = np.array([[0, 1, 0, 0] for _ in range(len(train_calling_images))]).reshape([-1, 4])
	train_normal_labels = np.array([[0, 0, 1, 0] for _ in range(len(train_normal_images))]).reshape([-1, 4])
	train_smoking_labels = np.array([[1, 0, 0, 0] for _ in range(len(train_smoking_images))]).reshape([-1, 4])
	train_smoking_images_labels = np.array([[0, 0, 0, 1] for _ in range(len(train_smoking_images_images))]).reshape([-1, 4])

	train_image_paths = np.concatenate([train_calling_images, train_normal_images, train_smoking_images, train_smoking_images_images])
	train_labels = np.concatenate([train_calling_labels, train_normal_labels, train_smoking_labels, train_smoking_images_labels])
	train_paths_labels = np.hstack((train_image_paths, train_labels))
	np.random.shuffle(train_paths_labels)

	val_calling_images = []
	val_normal_images = []
	val_smoking_images = []
	val_smoking_calling_images = []
	with open(val_txt, 'r') as f:
		val_lines = f.readlines()
	for line in val_lines:
		line = line.strip()
		if line.startswith('normal'):
			val_normal_images.append(os.path.join(train_dir, line))
		elif line.startswith('smoking_calling'):
			val_smoking_calling_images.append(os.path.join(train_dir, line))
		elif line.startswith('calling'):
			val_calling_images.append(os.path.join(train_dir, line))
		elif line.startswith('smoking'):
			val_smoking_images.append(os.path.join(train_dir, line))
	val_calling_images = np.array(val_calling_images).reshape([-1, 1])
	val_normal_images = np.array(val_normal_images).reshape([-1, 1])
	val_smoking_images = np.array(val_smoking_images).reshape([-1, 1])
	val_smoking_calling_images = np.array(val_smoking_calling_images).reshape([-1, 1])

	val_calling_labels = np.array([[0, 1, 0, 0] for _ in range(len(val_calling_images))]).reshape([-1, 4])
	val_normal_labels = np.array([[0, 0, 1, 0] for _ in range(len(val_normal_images))]).reshape([-1, 4])
	val_smoking_labels = np.array([[1, 0, 0, 0] for _ in range(len(val_smoking_images))]).reshape([-1, 4])
	val_smoking_calling_labels = np.array([[0, 0, 0, 1] for _ in range(len(val_smoking_calling_images))]).reshape([-1, 4])

	val_image_paths = np.concatenate([val_calling_images, val_normal_images, val_smoking_images, val_smoking_calling_images])
	val_labels = np.concatenate([val_calling_labels, val_normal_labels, val_smoking_labels, val_smoking_calling_labels])
	valdation_paths_labels = np.hstack((val_image_paths, val_labels))
	np.random.shuffle(valdation_paths_labels)

	print('total samples: %d, training samples: %d, validation samples: %d' % (
	len(train_paths_labels) + len(valdation_paths_labels), len(train_paths_labels), len(valdation_paths_labels)))

	train_dataset = BaseDataset(train_paths_labels, input_shape, 'train')
	validation_dataset = BaseDataset(valdation_paths_labels, input_shape, 'val')

	return train_dataset, validation_dataset