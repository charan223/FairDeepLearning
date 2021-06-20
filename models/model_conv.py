#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.model_mlp import MLP


"""Module for a single Convolutional Neural Network (CNN) unit
Parameters
----------
args: ArgumentParser
        Contains all model and shared input arguments
"""
class CNN(nn.Module):
    # Input to ConvVAE is resized to 32 * 32 * 3, each pixel has 3 float values
    # between 0, 1 to represent each of RGB channels
    def __init__(self, args):
        """Initializes CNN class"""
        super(CNN, self).__init__()

        self.z_dim = args.zdim
        self.batch_size = args.batch_size

        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(256, self.z_dim, 1)
        )

        self.weight_init()

    def weight_init(self, mode="normal"):
        """Initializes weights of CNN parameters"""
        for block in self._modules:
            for m in self._modules[block]:
                if type(m) == nn.Conv2d:
                    torch.nn.init.xavier_uniform_(m.weight)

    def encode(self, x):
        """Encodes an CNN input"""
        return self.enc(x).squeeze()


"""Module for CNN network
Parameters
----------
args: ArgumentParser
        Contains all model and shared input arguments
"""
class ConvNet(nn.Module):
    """Initializes CNN network and an MLP classifier"""
    def __init__(self, args):
        super(ConvNet, self).__init__()
        self.input_dim = args.input_dim
        self.sensattr = args.sensattr

        self.cnn = CNN(args)

        self.class_neurons = (
            [args.zdim] + args.cdepth * [args.cwidths] + [args.num_classes]
        )
        self.linear_layer = MLP(self.class_neurons)

        self.zdim = args.zdim
        self.n_sens = 1
        self.sens_idx = list(range(self.n_sens))
        self.nonsens_idx = [
            i for i in range(int(self.zdim / 2)) if i not in self.sens_idx
        ]
        self.count = 0

        self.optimizer = self.get_optimizer()

    @staticmethod
    def build_model(args):
        """ Builds ConvNet class """
        model = ConvNet(args)
        return model

    def all_params(self):
        """ Returns all parameters of the encoder CNN, classifier MLP """
        return list(self.cnn.parameters()) + list(self.linear_layer.parameters())

    def get_optimizer(self):
        """Returns optimizer over all parameters of the network"""
        return torch.optim.Adam(self.all_params())

    def forward(self, inputs, labels, attrs, mode="train", A_wts=None, Y_wts=None, AY_wts=None, reweight_target_tensor=None, reweight_attr_tensors=None):
        """Computes forward pass through encoder and classifier,
            Computes backward pass on the target function"""
        # Make inputs between 0, 1
        x = (inputs + 1) / 2

        # encode
        h_relu = self.cnn.encode(x)

        # classification mlp
        pre_softmax = self.linear_layer(h_relu)

        log_softmax = F.log_softmax(pre_softmax, dim=1)

        main_cost = F.nll_loss(log_softmax, labels)

        # This would be the target of optimization
        cost_dict = dict(
            main_cost=main_cost,  
        )
        if mode == "train":
            self.optimizer.zero_grad()
            main_cost.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params(), 5.0)
            self.optimizer.step()

        return pre_softmax, cost_dict
