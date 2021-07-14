from __future__ import print_function
import numpy as np
import collections


class Dataset:
    """
    Holds dataset attributes.
    This is a made-up dataset for debugging purposes.
    """

    def __init__(self, args):
        self.args = args
        self.device = self.args.device

        self.pos_label = 1  # e.g. required for TPR and sklearn.metrics.recall_score
        self.privileged_vals = {
            "sensitiveattr": 0
        }  # e.g. in Ricci dataset, Race is 'W' is considered priviledged
        self.sensitive_dict = {"sensitiveattr": list(range(args.num_groups))}
        self.read_data()
        self.AY_proportion = None

    def read_data(self):
        """This should define train_loader, valid_loader, test_loader, and inp_shape."""
        args = self.args
        device = self.device
        data_name = args.data
        if data_name == "clr-mnist":
            self.train_loader, self.valid_loader, self.test_loader = clr_mnist(
                args, device
            )
            self.inp_shape = (3, 32, 32)
        elif data_name == "adult":
            self.train_loader, self.valid_loader, self.test_loader = adult_dset(
                args, device
            )
            self.inp_shape = self.train_loader.input_shape
        else:
            raise ValueError("{} is not a valid dataset name.".format(data_name))

    def get_batch_iterator(
        self, which_set, batch_size, shuffle, keep_remainder=False, seed=None
    ):
        if self.args.data == "clr-mnist":
            if which_set == "train":
                x = self.train_loader.img
                y = self.train_loader.eo_lbl
                a = self.train_loader.att
            elif which_set == "valid":
                x = self.valid_loader.img
                y = self.valid_loader.eo_lbl
                a = self.valid_loader.att
            elif which_set == "test":
                x = self.test_loader.img
                y = self.test_loader.eo_lbl
                a = self.test_loader.att
            else:
                raise KeyError("Valid sets are: train, valid, and test.")
        elif self.args.data:
            if which_set == "train":
                x = self.train_loader.X
                y = self.train_loader.Y
                a = self.train_loader.A
            elif which_set == "valid":
                x = self.valid_loader.X
                y = self.valid_loader.Y
                a = self.valid_loader.A
            elif which_set == "test":
                x = self.test_loader.X
                y = self.test_loader.Y
                a = self.test_loader.A
            else:
                raise KeyError("Valid sets are: train, valid, and test.")
        else:
            raise KeyError("Wrong dataset name")

        sz = x.shape[0]
        batch_inds = make_batch_inds(sz, batch_size, shuffle, keep_remainder, seed)
        iterator = DatasetIterator([x, y, a], batch_inds)
        return iterator

    def get_A_proportions(self):
        AY = self.get_AY_proportions()
        ret = [AY[0][0] + AY[0][1], AY[1][0] + AY[1][1]]
        assert np.sum(ret) == 1.0
        return ret

    def get_Y_proportions(self):
        AY = self.get_AY_proportions()
        ret = [AY[0][0] + AY[1][0], AY[0][1] + AY[1][1]]
        assert np.sum(ret) == 1.0
        return ret

    def get_AY_proportions(self):
        if self.AY_proportion:
            return self.AY_proportion
        # Compute if None
        if self.args.data == "clr-mnist":
            A = self.train_loader.att
            Y = self.train_loader.eo_lbl
            ttl = len(self.train_loader.att)
        elif self.args.data == "adult":
            A = self.train_loader.A
            Y = self.train_loader.Y
            ttl = len(self.train_loader.A)
        len_A0Y0 = len([ay for ay in zip(A, Y) if ay == (0, 0)])
        len_A0Y1 = len([ay for ay in zip(A, Y) if ay == (0, 1)])
        len_A1Y0 = len([ay for ay in zip(A, Y) if ay == (1, 0)])
        len_A1Y1 = len([ay for ay in zip(A, Y) if ay == (1, 1)])

        assert (
            len_A0Y0 + len_A0Y1 + len_A1Y0 + len_A1Y1
        ) == ttl, "Problem computing train set AY proportion."
        A0Y0 = len_A0Y0 / ttl
        A0Y1 = len_A0Y1 / ttl
        A1Y0 = len_A1Y0 / ttl
        A1Y1 = len_A1Y1 / ttl
        # print(f'A0Y0 = {A0Y0}, A0Y1 = {A0Y1}, A1Y0 = {A1Y0}, A1Y1 = {A1Y1}')
        self.AY_proportion = [[A0Y0, A0Y1], [A1Y0, A1Y1]]
        # print(f'total sum = {np.sum(self.AY_proportion)}')
        # assert np.sum(self.AY_proportion) == 1., 'Problem computing train set AY proportion.'
        return self.AY_proportion


def adult_dset(args, device):
    from dataloaders.adult_loader import AdultDataset

    train_loader = AdultDataset(
        phase="train",
        root_dir=args.data_dir,
        tar_attr="income",
        priv_attr=args.sensattr,
        clr_ratio=[args.beta_1, args.beta_2],
    )
    val_loader = AdultDataset(
        phase="val",
        root_dir=args.data_dir,
        tar_attr="income",
        priv_attr=args.sensattr,
        clr_ratio=[args.beta_1, args.beta_2],
    )
    test_loader = AdultDataset(
        phase="test",
        root_dir=args.data_dir,
        tar_attr="income",
        priv_attr=args.sensattr,
        clr_ratio=[args.beta_1, args.beta_2],
    )

    return train_loader, val_loader, test_loader


def clr_mnist(args, device):
    from dataloaders.mnist_loader import MNISTthin

    train_loader = MNISTthin(
        which_set="train",
        path=args.data_dir,
        background=args.bck,
        label_type=args.label_type,
        clr_ratio=[args.beta_1, args.beta_2],
        shade=args.useshade,
        green_yellow=args.green_yellow,
        egr=args.egr,
        ogr=args.ogr,
        sensitiveattr=args.sensattr,
        num_groups=args.num_groups,
        transform_list=[],
        out_channels=3,
    )
    valid_loader = MNISTthin(
        which_set="valid",
        path=args.data_dir,
        background=args.bck,
        label_type=args.label_type,
        clr_ratio=[args.beta_1, args.beta_2],
        shade=args.useshade,
        green_yellow=args.green_yellow,
        egr=args.egr,
        ogr=args.ogr,
        sensitiveattr=args.sensattr,
        num_groups=args.num_groups,
        transform_list=[],
        out_channels=3,
    )
    test_loader = MNISTthin(
        which_set="test",
        path=args.data_dir,
        background=args.bck,
        label_type=args.label_type,
        clr_ratio=[0.5, 0.5],
        shade=args.useshade,
        green_yellow=args.green_yellow,
        egr=0.5,
        ogr=0.5,
        sensitiveattr=args.sensattr,
        num_groups=args.num_groups,
        transform_list=[],
        out_channels=3,
    )
    return train_loader, valid_loader, test_loader


def make_batch_inds(n, batch_size, shuffle, keep_remainder=False, seed=None):
    if seed:  # if already set, do not use this
        np.random.seed(seed)
    if not keep_remainder:
        n = (n // batch_size) * batch_size  # round to nearest mb_size
    # shuffle every time differently if seed not set.
    shuf = np.random.permutation(n) if shuffle else np.arange(n)
    return np.array_split(shuf, n // batch_size)


class DatasetIterator(collections.Iterator):
    def __init__(self, tensor_list, ind_list):
        self.tensors = tensor_list
        self.inds = ind_list
        self.curr = 0
        self.ttl_minibatches = len(self.inds)

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr >= self.ttl_minibatches:
            raise StopIteration
        else:
            inds = self.inds[self.curr]
            minibatch = [t[inds] for t in self.tensors]
            self.curr += 1
            return minibatch
