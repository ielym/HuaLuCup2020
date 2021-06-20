class StepLR:
    def __init__(self, optimizer, factor, patience, min_lr, **kwargs):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.kwargs = kwargs

        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

    def step(self, **kwargs):
        history = kwargs['history']
        epoch = kwargs['epoch']

        for param_group in self.optimizer.param_groups:
            lr = param_group['initial_lr'] * (self.factor ** (epoch // self.patience))
            param_group['lr'] = lr if lr > self.min_lr else self.min_lr

class StepLRAfter:
    def __init__(self, optimizer, factor, patience, after, min_lr, **kwargs):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.kwargs = kwargs
        self.after = after

        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

    def step(self, **kwargs):
        history = kwargs['history']
        epoch = kwargs['epoch']

        if epoch > self.after:
            for param_group in self.optimizer.param_groups:
                lr = param_group['initial_lr'] * (self.factor ** (epoch // self.patience))
                param_group['lr'] = lr if lr > self.min_lr else self.min_lr
