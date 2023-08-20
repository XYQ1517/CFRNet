import math


class LR_Scheduler(object):
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=20, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.old_lr = self.lr

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'loss_lr_1':
            lr = self.old_lr / 5
            self.old_lr = lr
        elif self.mode == 'loss_lr_2':
            lr = self.old_lr / 2
            self.old_lr = lr
        # elif self.mode == 'cos':
        #     lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        # elif self.mode == 'poly':
        #     lr = self.lr * pow((1 - 1.0 * T / self.N), 3)
        # elif self.mode == 'step':
        #     lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.8f, \
                previous best IoU= %.6f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)
        return lr

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10
