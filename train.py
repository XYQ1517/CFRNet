import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.loss import Loss
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from modeling.dinknet import DinkNet34
from modeling.unet import Unet
from modeling.MAResUnet import MAResUNet
from modeling.NLLinkNet import NL34_LinkNet
from modeling.Segformer import Segformer
from modeling.DBRANet import DBRANet_8
from CFRNet import CFRNet


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        # model = CFRNet(image_size=(args.image_size, args.image_size))
        model = NL34_LinkNet()

        # Define Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        loss = Loss(cuda=True)
        self.criterion = loss.build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(2)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_mIoU = 0.0
        self.best_IoU = 0.0
        self.best_Precision = 0.0
        self.best_Recall = 0.0
        self.best_F1 = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_mIoU = checkpoint['best_mIoU']
            self.best_IoU = checkpoint['best_IoU']
            self.best_Precision = checkpoint['best_Precision']
            self.best_Recall = checkpoint['best_Recall']
            self.best_F1 = checkpoint['best_F1']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        self.loss_number = 0
        self.factor = False
        self.lr = args.lr

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            if self.factor:
                self.lr = self.scheduler(self.optimizer, i, epoch, self.best_IoU)
                self.factor = False
            self.optimizer.zero_grad()
            output = self.model(image)
            target = torch.unsqueeze(target, 1)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.5f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            pred = output.data.cpu().numpy()
            target_n = target.cpu().numpy()
            # Add batch sample into evaluator
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            self.evaluator.add_batch(target_n, pred)

        train_loss /= num_img_tr
        PA = self.evaluator.Pixel_Accuracy()
        MPA = self.evaluator.Mean_Pixel_Accuracy()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train/PA', PA, epoch)
        self.writer.add_scalar('train/MPA', MPA, epoch)
        self.writer.add_scalar('train/mIoU', mIoU, epoch)
        self.writer.add_scalar('train/IoU', IoU, epoch)
        self.writer.add_scalar('train/Precision', Precision, epoch)
        self.writer.add_scalar('train/Recall', Recall, epoch)
        self.writer.add_scalar('train/F1', F1, epoch)
        print('Train:')
        print('[Epoch:{}, Train loss:{}]'.format(epoch, train_loss))
        print("PA:{}, MPA:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
              .format(PA, MPA, mIoU, IoU, Precision, Recall, F1))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_IoU': self.best_IoU,
            }, is_best)
        return train_loss

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        num_img_tr = len(self.val_loader)
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            target = torch.unsqueeze(target, 1)
            loss = self.criterion(output, target)
            val_loss += loss.item()
            tbar.set_description('Test loss: %.5f' % (val_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target_n = target.cpu().numpy()
            # Add batch sample into evaluator
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            self.evaluator.add_batch(target_n, pred)

            if i % (num_img_tr // 1) == 0:
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, i, split='Val')

        # Fast test during the training
        val_loss /= num_img_tr
        PA = self.evaluator.Pixel_Accuracy()
        MPA = self.evaluator.Mean_Pixel_Accuracy()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()
        self.writer.add_scalar('val/total_loss_epoch', val_loss, epoch)
        self.writer.add_scalar('val/PA', PA, epoch)
        self.writer.add_scalar('val/MPA', MPA, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/IoU', IoU, epoch)
        self.writer.add_scalar('val/Precision', Precision, epoch)
        self.writer.add_scalar('val/Recall', Recall, epoch)
        self.writer.add_scalar('val/F1', F1, epoch)
        print('Validation:')
        print('[Epoch:{}, Val loss:{}]'.format(epoch, val_loss))
        print("PA:{}, MPA:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
              .format(PA, MPA, mIoU, IoU, Precision, Recall, F1))

        new_IoU = IoU
        if new_IoU > self.best_IoU:
            is_best = True
            self.best_IoU = new_IoU
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_IoU': self.best_IoU,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='DeepGlobe',
                        choices=['DeepGlobe', 'IRDS', 'CHN6'],
                        help='dataset name (default: DeepGlobe)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--image_size', type=int, default=512,
                        choices=[1024, 512, 256],
                        help='image size')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='BCE_Dice',
                        choices=['ce', 'con_ce', 'focal', 'BCE_Dice', 'Focal_Dice', 'PerceptualLoss'],
                        help='loss func type')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:1)')
    parser.add_argument('--batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                training (default: 8)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        choices=[5e-4, 3e-4, 2e-4],
                        help='CFRNet:5e-4')
    parser.add_argument('--lr-scheduler', type=str, default='loss_lr_1',
                        choices=['poly', 'step', 'cos', 'loss_lr_1', 'loss_lr_2'],
                        help='CFRNet:loss_lr_2')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must bomma-separated list of integers only')

    if args.checkname is None:
        args.checkname = 'NL34_LinkNet'
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    train_best_loss = 10000
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        train_loss = trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
        if train_loss <= train_best_loss:
            train_best_loss = train_loss
            trainer.loss_number = 0
        else:
            trainer.loss_number += 1
        if trainer.loss_number > 3:
            if trainer.lr < 5e-7:
                break
            trainer.factor = True
    print("Finish!")
    trainer.writer.close()


if __name__ == "__main__":
   main()
