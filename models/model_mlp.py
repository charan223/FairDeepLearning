#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


"""Module for a single Multi Layer Perception (MLP) unit
Parameters
----------
num_neurons: list, dtype: int
        Number of neurons in each layer
activ: string, default: "leakyrelu"
        Activation function
"""
class MLP(nn.Module):
    def __init__(self, num_neurons, activ="leakyrelu"):
        """Initializes MLP unit"""
        super(MLP, self).__init__()
        self.num_neurons = num_neurons
        self.num_layers = len(self.num_neurons) - 1
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                for i in range(self.num_layers)
            ]
        )
        for hidden in self.hiddens:
            torch.nn.init.xavier_uniform_(hidden.weight)
        self.activ = activ

    def forward(self, inputs):
        """Computes forward pass through the model"""
        L = inputs
        for hidden in self.hiddens:
            L = hidden(L)
            if self.activ == "softplus":
                L = F.softplus(L)
            elif self.activ == "sigmoid":
                L = F.sigmoid(L)
            elif self.activ == "relu":
                L = F.relu(L)
            elif self.activ == "leakyrelu":
                L = F.leaky_relu(L)
            elif self.activ == "None":
                pass
            else:
                raise Exception("bad activation function")
        return L

    def freeze(self):
        """Stops gradient computation through MLP parameters"""
        for para in self.parameters():
            para.requires_grad = False

    def activate(self):
        """Activates gradient computation through MLP parameters"""
        for para in self.parameters():
            para.requires_grad = True



"""Module for MLP network
Parameters
----------
args: ArgumentParser
        Contains all model and shared input arguments
"""
class MLPNet(nn.Module):
    def __init__(self, args):
        super(MLPNet, self).__init__()
        self.input_dim = args.input_dim

        self.enc_neurons = [self.input_dim] + args.edepth * [args.ewidths] + [args.zdim]
        self.encoder = MLP(self.enc_neurons)

        self.class_neurons = (
            [args.zdim] + args.cdepth * [args.cwidths] + [args.num_classes]
        )
        self.classifier = MLP(self.class_neurons)

        # Parameter of the final softmax classification layer.
        self.num_classes = args.num_classes

        self.optimizer = self.get_optimizer()

    @staticmethod
    def build_model(args):
        """ Builds MLPNet class """
        model = MLPNet(args)
        return model

    def all_params(self):
        """ Returns all parameters of the MLP model """
        return list(self.encoder.parameters()) + list(self.classifier.parameters())

    def get_optimizer(self):
        """Returns optimizer over all parameters of MLP"""
        return torch.optim.Adam(self.all_params())

    def forward(self, inputs, labels, attrs, mode="train", A_wts=None, Y_wts=None, AY_wts=None, reweight_target_tensor=None, reweight_attr_tensors=None):
        """Computes forward pass through encoder and classifier,
            Computes backward pass on the target function"""
        h_relu = inputs

        # encoder mlp
        h_relu = self.encoder(h_relu)

        # classification mlp
        pre_softmax = self.classifier(h_relu)

        log_softmax = F.log_softmax(pre_softmax, dim=1)

        main_cost = F.nll_loss(log_softmax, labels)

        # target of optimization
        cost_dict = dict(
            main_cost=main_cost,  
        )
        if mode == "train":
            self.optimizer.zero_grad()
            main_cost.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params(), 5.0)
            self.optimizer.step()

        return pre_softmax, cost_dict
