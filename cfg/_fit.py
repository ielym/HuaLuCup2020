import torch
from torchtoolbox.tools import mixup_data, mixup_criterion

import time
from ._history import History
import sys

def _register_history(criterion, metrics):
	history = History()

	for k, v in criterion.items():
		name = k
		val_name= 'val_{}'.format(name)

		history.register(name, ':.4e')
		history.register(val_name, ':.4e')

	for k, v in metrics.items():
		name = k
		val_name= 'val_{}'.format(name)

		history.register(name, ':6.3f')
		history.register(val_name, ':6.3f')

	return history

def Fit(data_flow, model, args, batch_size, optimizer, criterion, metrics, reduce_lr, checkpoint, verbose, workers):

	history = _register_history(criterion, metrics)

	for epoch in range(args.start_epoch, args.max_epochs):
		train_dataset, validation_dataset = data_flow(base_dir=args.data_local, input_shape=(args.img_channel, args.input_size, args.input_size))
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
		validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

		reduce_lr.step(history=history, epoch=epoch)
		print('\nEpoch {} learning_rate : {}'.format(epoch+1, optimizer.param_groups[0]['lr']))

		_Train(model, history, train_loader, optimizer, criterion, metrics, verbose, epoch)
		_Validate(model, history, validation_loader, criterion, metrics, epoch)

		for ckpt in checkpoint:
			ckpt.savemodel(model, epoch=epoch+1, **history.history_dict)


def _Train(model, history, train_loader, optimizer, criterion, metrics, verbose, epoch):

	model.train()

	history.reset()
	s_time = time.time()
	total_batch = len(train_loader)

	for batch, (images, target) in enumerate(train_loader):
		history.update(n=1, name='DataTime', val=time.time() - s_time)

		images = images.cuda(non_blocking=True)
		y_true = target.cuda(non_blocking=True)

		images, labels_a, labels_b, lam = mixup_data(images, y_true, 0.2)

		y_pre = model(images)


		metric1 = metrics['acc@1'](y_pre, torch.argmax(y_true, dim=1).unsqueeze(dim=1).squeeze())
		history.update(n=images.shape[0], name='acc@1', val=metric1.item())
		metric5 = metrics['acc@3'](y_pre, torch.argmax(y_true, dim=1).unsqueeze(dim=1).squeeze(), topk=2)
		history.update(n=images.shape[0], name='acc@3', val=metric5.item())

		optimizer.zero_grad()
		# loss1 = criterion['lossL'](y_pre, torch.argmax(y_true, dim=1).unsqueeze(dim=1).squeeze())
		loss1 = mixup_criterion(criterion['lossL'], y_pre, torch.argmax(labels_a, dim=1).unsqueeze(dim=1).squeeze(), torch.argmax(labels_b, dim=1).unsqueeze(dim=1).squeeze(), lam)
		history.update(n=images.shape[0], name='lossL', val=loss1.item())
		# loss2 = criterion['lossS'](y_pre, torch.argmax(y_true, dim=1).unsqueeze(dim=1).squeeze())
		loss2 = mixup_criterion(criterion['lossS'], y_pre, torch.argmax(labels_a, dim=1).unsqueeze(dim=1).squeeze(), torch.argmax(labels_b, dim=1).unsqueeze(dim=1).squeeze(), lam)
		history.update(n=images.shape[0], name='lossS', val=loss2.item())
		loss1.backward(retain_graph=True)
		loss2.backward()
		optimizer.step()

		history.update(n=1, name='BatchTime', val=time.time() - s_time)

		if (batch) % verbose == 0:
			history.display(epoch+1, batch+1, total_batch, 'train')
		s_time = time.time()

def _Validate(model, history, val_loader, criterion, metrics, epoch):
	model.eval()
	with torch.no_grad():
		for batch, (images, target) in enumerate(val_loader):

			images = images.cuda(non_blocking=True)
			y_true = target.cuda(non_blocking=True)

			y_pre = model(images)

			metric1 = metrics['acc@1'](y_pre, torch.argmax(y_true, dim=1).unsqueeze(dim=1).squeeze())
			history.update(n=images.shape[0], name='val_acc@1', val=metric1.item())
			metric5 = metrics['acc@3'](y_pre, torch.argmax(y_true, dim=1).unsqueeze(dim=1).squeeze(), topk=2)
			history.update(n=images.shape[0], name='val_acc@3', val=metric5.item())

			loss1 = criterion['lossL'](y_pre, torch.argmax(y_true, dim=1).unsqueeze(dim=1).squeeze())
			history.update(n=images.shape[0], name='val_lossL', val=loss1.item())
			loss2 = criterion['lossS'](y_pre, torch.argmax(y_true, dim=1).unsqueeze(dim=1).squeeze())
			history.update(n=images.shape[0], name='val_lossS', val=loss2.item())

		history.display(epoch=epoch+1, mode='validate')