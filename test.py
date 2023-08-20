import argparse
import os
from dataloaders.utils import *
from torchvision.utils import make_grid  # save_image
from dataloaders import make_data_loader
from utils.metrics import Evaluator
from tqdm import tqdm
from modeling.unet import Unet
from modeling.dinknet import DinkNet34
from modeling.Segformer import Segformer
from modeling.DBRANet import DBRANet_4
from modeling.HCTNet import HCTNet
from modeling.MAResUnet import MAResUNet
from modeling.NLLinkNet import NL34_LinkNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((256, 256))
    im.save(filename)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--out-path', type=str, default='./run/CHN6/',
                        help='mask image to save')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for test ')
    parser.add_argument('--ckpt', type=str, default='./run/CHN6/DinkNet34/model_best.pth.tar',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--loss-type', type=str, default='con_ce',
                        choices=['ce', 'con_ce', 'focal'],
                        help='loss func type')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='CHN6',
                        choices=['DeepGlobe', 'IRDS', 'CHN6'],
                        help='dataset name')
    parser.add_argument('--image-size', type=int, default=512,
                        help='base image size. DeepGlobe:1024.')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    kwargs = {'num_workers': args.workers, 'pin_memory': False}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)

    # model = HCTNet(image_size=(args.image_size, args.image_size))
    model = DinkNet34()
    model = model.cuda()
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['state_dict'])

    out_path = os.path.join(args.out_path, 'Output', 'DinkNet34/')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    evaluator = Evaluator(2)
    model.eval()
    evaluator.reset()
    tbar = tqdm(test_loader, desc='\r')
    for i, sample in enumerate(tbar):
        image, target = sample[0]['image'], sample[0]['label']
        img_name = sample[1][0].split('.')[0]
        if args.cuda:
            image = image.cuda()
        # with torch.no_grad():
        #     output = predict(image, model)
        output = model(image).squeeze().cpu().data.numpy()
        target_n = target.cpu().numpy()
        output[output < 0.5] = 0
        output[output >= 0.5] = 1
        output = np.expand_dims(output, axis=0)
        mask = output * 255

        evaluator.add_batch(target_n, output.astype(int))

        # save imgs
        out_image = make_grid(image[0, :].clone().cpu().data, 3, normalize=True)
        out_GT = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=args.dataset), 3, normalize=False, range=(0, 255))
        out_pred_label_sum = make_grid(decode_seg_map_sequence(mask,
                                                       dataset=args.dataset), 3, normalize=False, range=(0, 255))

        save_image(out_image, out_path + img_name + '_sat.png')
        save_image(out_GT, out_path + img_name + '_GT' + '.png')
        save_image(out_pred_label_sum, out_path + img_name + '_pred' + '.png')

    # Fast test during the training
    PA = evaluator.Pixel_Accuracy()
    MPA = evaluator.Mean_Pixel_Accuracy()
    mIoU = evaluator.Mean_Intersection_over_Union()
    IoU = evaluator.Intersection_over_Union()
    Precision = evaluator.Pixel_Precision()
    Recall = evaluator.Pixel_Recall()
    F1 = evaluator.Pixel_F1()
    print('Test:')
    print("PA:{}, MPA:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
          .format(PA, MPA, mIoU, IoU, Precision, Recall, F1))


if __name__ == "__main__":
   main()
