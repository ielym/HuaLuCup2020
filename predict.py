# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import multiprocessing
import random
import torch.backends.cudnn as cudnn
import time
from torchsummary import summary
import cv2

from data_gen_predict import data_flow
from models.model import ResNet101, ResNet50, EfficientB7
from utils import RESULT_KINDS, RESULT_KINDS_ENGLISH, scale_img_minsize, scale_img_maxsize, showTorchImg

def model_fn(args):
	model1 = ResNet101(weights=None, input_shape=(args.img_channel, 224, 224), num_classes=args.num_classes)
	pretrained_dict = torch.load(args.inference_weights1)
	single_dict = {}
	for k, v in pretrained_dict.items():
		single_dict[k[7:]] = v
	model1.load_state_dict(single_dict)
	model1 = nn.DataParallel(model1)
	model1 = model1.cuda()
	# =======================================================================================================================
	model2 = ResNet50(weights=None, input_shape=(args.img_channel, 224, 224), num_classes=args.num_classes)
	pretrained_dict = torch.load(args.inference_weights2)
	single_dict = {}
	for k, v in pretrained_dict.items():
		single_dict[k[7:]] = v
	model2.load_state_dict(single_dict)
	model2 = model2.cuda()
	# =======================================================================================================================
	model3 = EfficientB7(weights=None, input_shape=(args.img_channel, 224, 224), num_classes=args.num_classes)
	pretrained_dict = torch.load(args.inference_weights3)
	single_dict = {}
	for k, v in pretrained_dict.items():
		single_dict[k[7:]] = v
	model3.load_state_dict(single_dict)
	model3 = model3.cuda()
	return model1, model2, model3

def get_result(output):
	output = torch.softmax(output, dim=1)
	output = output.cpu().numpy()
	return output

def submit(args):
	model1, model2, model3 = model_fn(args)

	batch_size = args.batch_size
	test_dataset, img_files = data_flow(args.data_local, h_w=1.4, max_size=[425, 655, 600])
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True, drop_last=False)

	predict = np.array([[0.0, 0.0] for _ in range(len(img_files))], dtype=np.float).reshape([-1, 2])

	results = []
	already = 0
	model1.eval()
	model2.eval()
	model3.eval()
	with torch.no_grad():
		for batch, (images1, images2, images3) in enumerate(test_loader):
			images1 = images1.cuda(non_blocking=True)
			images2 = images2.cuda(non_blocking=True)
			images3 = images3.cuda(non_blocking=True)

			b = images1.size()[0]

			output1 = model1(images1)
			output1 = get_result(output1)

			output2 = model2(images2)
			output2 = get_result(output2)

			output3 = model3(images3)
			output3 = get_result(output3)

			# 9631578947368421 - 957363237704545 - 425-655-600
			scores = np.mean([output1, output2, output3], axis=0)
			kinds = np.argmax(scores, axis=1)
			scores = np.max(scores, axis=1)

			predict[batch*batch_size : batch*batch_size + b, 0] = kinds
			predict[batch*batch_size : batch*batch_size + b, 1] = scores
			already += b
			print('\r {}'.format(already), end='')
			if already == len(img_files):
				break

	for idx in range(len(img_files)):
		image_name = os.path.split(str(img_files[idx, 0]))[-1]

		category_id = predict[idx, 0]
		vote_kind_english = RESULT_KINDS_ENGLISH[int(category_id)]

		category_score = predict[idx, 1]

		results.append({"image_name":image_name, "category":vote_kind_english, "score":category_score},)
	with open('/notebooks/results/result.json', 'w') as f:
		f.write(str(results))

def _predict(args):
	print('Test on weight1 (ResNext101) : ', os.path.split(args.inference_weights1)[-1])
	print('Test on weight2 (ResNet50) : ', os.path.split(args.inference_weights2)[-1])
	print('Test on weight3 (Efficient-B7) : ', os.path.split(args.inference_weights3)[-1])
	submit(args)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='PyTorch Training')
	parser.add_argument('--data_local', default=r'S:\tempsets', type=str, help='')
	parser.add_argument('--input_size', default=400, type=int, help='')
	parser.add_argument('--img_channel', default=3, type=int, help='')
	parser.add_argument('--num_classes', default=4, type=int, help='')
	parser.add_argument('--batch_size', default=8, type=int, help='')

	parser.add_argument('--inference_weights1', default='./models/best_resnext101.pth', type=str, help='')
	parser.add_argument('--inference_weights2', default='./models/best_resnet50.pth', type=str, help='')
	parser.add_argument('--inference_weights3', default='./models/best_efficientb7.pth', type=str, help='')

	args, unknown = parser.parse_known_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = '0'

	if not os.path.exists(args.data_local):
		raise Exception('FLAGS.data_local_path: %s is not exist' % args.data_local)

	_predict(args)


