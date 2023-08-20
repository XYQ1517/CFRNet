class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'IRDS':
            return 'data/Istanbul_Road_Data_Set/'
        elif dataset == 'DeepGlobe':
            return 'data/deepglobe/'
        elif dataset == 'CHN6':
            return 'data/CHN6/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
