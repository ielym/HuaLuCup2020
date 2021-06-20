# -*- coding: utf-8 -*-
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import os
import random

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['FangSong']  #可显示中文字符
plt.rcParams['axes.unicode_minus']=False

RESULT_KINDS = {
				'smoking':0,
				'calling':1,
				'normal': 2,
				'smoking_calling':3,
			}
RESULT_KINDS_ENGLISH = {
	0:'smoking',
	1:'calling',
	2: 'normal',
	3:'smoking_calling',
}


def scale_img_minsize(ori_h, ori_w, minsize=600):
	if ori_w > ori_h:
		target_h = minsize
		target_w = 1.0 * ori_w * target_h / ori_h
	else:
		target_w = minsize
		target_h = 1.0 * ori_h * target_w / ori_w

	target_h = int(target_h)
	target_w = int(target_w)

	scale_h = 1.0 * target_h / ori_h
	scale_w = 1.0 * target_w / ori_w

	return target_h, target_w, scale_h, scale_w

def scale_img_maxsize(ori_h, ori_w, maxsize=600):
	if ori_w > ori_h:
		target_w = maxsize
		target_h = 1.0 * ori_h * target_w / ori_w
	else:
		target_h = maxsize
		target_w = 1.0 * ori_w * target_h / ori_h

	target_h = int(target_h)
	target_w = int(target_w)

	scale_h = 1.0 * target_h / ori_h
	scale_w = 1.0 * target_w / ori_w

	return target_h, target_w, scale_h, scale_w

def showTorchImg(img):
	img = img.numpy()
	img = np.transpose(img, [1,2,0])
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	cv2.imshow('img', img)
	cv2.waitKey()

class AddPepperNoise(object):
	def __init__(self, snr, p=0.9):
		assert isinstance(snr, float) or (isinstance(p, float))
		self.snr = snr
		self.p = p

	def __call__(self, img):
		if random.uniform(0, 1) < self.p:
			img_ = np.array(img).copy()
			h, w, c = img_.shape
			signal_pct = self.snr
			noise_pct = (1 - self.snr)
			mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
			mask = np.repeat(mask, c, axis=2)
			img_[mask == 1] = 255   # 盐噪声
			img_[mask == 2] = 0     # 椒噪声
			return Image.fromarray(img_.astype('uint8')).convert('RGB')
		else:
			return img

class AddGauseNoise(object):
	def __init__(self, sigma=None, p=0.9):
		assert isinstance(sigma, float) or (isinstance(p, float))
		self.sigma = sigma
		self.p = p

	def __call__(self, img):
		if self.sigma == None:
			self.sigma = np.random.randint(5, 25)
			# self.sigma = np.random.choice([15,25,35,50,75])

		if random.uniform(0, 1) < self.p:
			img_ = np.array(img).copy()
			noise = np.random.normal(0., self.sigma, img_.shape)
			img_ = img_ + noise
			return Image.fromarray(img_.astype('uint8')).convert('RGB')
		else:
			return img