from dataloaders.datasets import deepglobe, IRDS, CHN6
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def make_data_loader(args, **kwargs):

    if args.dataset == 'DeepGlobe':
        train_set = deepglobe.Segmentation(args, split='train')
        val_set = deepglobe.Segmentation(args, split='val')
        test_set = deepglobe.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'IRDS':
        train_set = IRDS.Segmentation(args, split='train')
        val_set = IRDS.Segmentation(args, split='val')
        test_set = IRDS.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'CHN6':
        train_set = CHN6.Segmentation(args, split='train')
        val_set = CHN6.Segmentation(args, split='val')
        test_set = CHN6.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

