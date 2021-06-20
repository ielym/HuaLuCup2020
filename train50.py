# -*- coding: utf-8 -*-
import argparse
import random
import numpy as np
import torch.backends.cudnn as cudnn
import multiprocessing
import time
import torch
import torch.nn as nn
from torchsummary import summary
import os

from cfg import _metrics, _fit, _modelcheckpoint, _reducelr, _criterion
from data_gen_train import data_flow
from models.model import ResNet50, EfficientB7, ResNet101

def model_fn(args, mode):
	model = ResNet50(weights=args.pretrained_weights, input_shape=(args.img_channel, args.input_size, args.input_size), num_classes=args.num_classes)

	for param in model.parameters():
		param.requires_grad = True

	for name, value in model.named_parameters():
		print(name, value.requires_grad)

	model = nn.DataParallel(model)
	model = model.cuda()
	return model

def train_model(args, mode):
	model = model_fn(args, mode)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	criterion = {'lossL' : nn.CrossEntropyLoss().cuda(), 'lossS' : _criterion.LabelSmoothSoftmaxCE().cuda()}
	metrics = {"acc@1" : _metrics.top1_accuracy, "acc@3" : _metrics.topk_accuracy}
	checkpoint1 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'best_resnet50.pth'), monitor='val_acc@1', mode='max', verbose=1, save_best_only=True, save_weights_only=True)
	checkpoint2 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'ep{epoch:05d}-val_acc@1_{val_acc@1:.4f}-val_lossS_{val_lossS:.4f}-val_lossL_{val_lossL:.4f}.pth'), monitor='val_acc@1', mode='max', verbose=1, save_best_only=True, save_weights_only=True)
	reduce_lr = _reducelr.StepLR(optimizer, factor=0.2, patience=8, min_lr=1e-6)
	_fit.Fit(
			data_flow = data_flow,
			model=model,
			args=args,
			batch_size = args.batch_size,
			optimizer=optimizer,
			criterion=criterion,
			metrics=metrics,
			reduce_lr = reduce_lr,
			checkpoint = [checkpoint1, checkpoint2],
			verbose=1,
			workers=int(multiprocessing.cpu_count() * 0.8),
		)
	print('Training Done!')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PyTorch Training')
	parser.add_argument('--data_local', default=r'/notebooks', type=str, help='')
	parser.add_argument('--input_size', default=400, type=int, help='')
	parser.add_argument('--img_channel', default=3, type=int, help='')
	parser.add_argument('--num_classes', default=4, type=int, help='')
	parser.add_argument('--batch_size', default=32, type=int, help='')
	parser.add_argument('--learning_rate', default=1e-4, type=float, help='')
	parser.add_argument('--max_epochs', default=40, type=int, help='')
	parser.add_argument('--start_epoch', default=0, type=int, help='')
	parser.add_argument('--pretrained_weights', default='./models/zoo/resnet50-19c8e357.pth', type=str, help='')
	parser.add_argument('--seed', default=None, type=int, help='')

	args, unknown = parser.parse_known_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4, 5, 6, 7, 8'
	print('CUDA device count : {}'.format(torch.cuda.device_count()))

	if not os.path.exists(args.data_local):
		raise Exception('FLAGS.data_local_path: %s is not exist' % args.data_local)

	if args.seed != None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		np.random.seed(args.seed)
		cudnn.deterministic = True
		print('You have chosen to seed training with seed {}.'.format(args.seed))
	else:
		print('You have chosen to random seed.')

	train_model(args=args, mode='train')

