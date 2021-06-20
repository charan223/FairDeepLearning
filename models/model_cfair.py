#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.model_mlp import MLP

class GradReverse(Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


"""Module for CFAIR network
Parameters
----------
args: ArgumentParser
        Contains all model and shared input arguments
"""
class CFairNet(nn.Module):
    """Initializes CFAIR network: MLP encoder, MLP classifier, 2 MLP discriminators"""
    def __init__(self, args):
        super(CFairNet, self).__init__()
        self.input_dim = args.input_dim
        self.num_classes = args.num_classes

        self.enc_neurons = [self.input_dim] + args.edepth * [args.ewidths] + [args.zdim]
        self.encoder = MLP(self.enc_neurons)

        self.class_neurons = (
            [args.zdim] + args.cdepth * [args.cwidths] + [args.num_classes]
        )
        self.classifier = MLP(self.class_neurons)

        self.adv_neurons = (
            [args.zdim] + args.adepth * [args.awidths] + [args.num_groups]
        )
        self.discriminator = nn.ModuleList(
            [MLP(self.adv_neurons) for _ in range(self.num_classes)]
        )

        self.adv_coeff = args.adv_coeff
        self.device = args.device
        self.data = args.data

        self.optimizer = self.get_optimizer()

    @staticmethod
    def build_model(args):
        """ Builds CFAIR network """
        model = CFairNet(args)
        return model

    def all_params(self):
        """Returns all model parameters"""
        return (
            list(self.encoder.parameters())
            + list(self.classifier.parameters())
            + list(self.discriminator.parameters())
        )

    def get_optimizer(self):
        """Returns an optimizer"""
        if self.data == "adult":
            optim = torch.optim.Adadelta(self.all_params())
        else:
            optim = torch.optim.Adam(self.all_params())
        return optim

    def forward(self, inputs, labels, attrs, mode="train", A_wts=None, Y_wts=None, AY_wts=None, reweight_target_tensor=None, reweight_attr_tensors=None):
        """Computes forward pass through encoder ,
            Computes backward pass on the target function"""
        h_relu = inputs

        h_relu = self.encoder(h_relu)

        # Classification probabilities.
        pre_softmax = self.classifier(h_relu)
        logprobs = F.log_softmax(pre_softmax, dim=1)

        main_cost = F.nll_loss(
            logprobs,
            labels,
            weight=reweight_target_tensor.type(torch.FloatTensor).to(self.device),
        )

        # Adversary component.
        c_losses = []
        h_relu = grad_reverse(h_relu)
        for j in range(self.num_classes):
            idx = labels == j
            c_h_relu = h_relu[idx]
            c_cls = F.log_softmax(self.discriminator[j](c_h_relu), dim=1)
            c_losses.append(c_cls)

        adv_cost = torch.mean(
            torch.stack(
                [
                    F.nll_loss(
                        c_losses[j],
                        attrs[labels == j],
                        weight=reweight_attr_tensors[j]
                        .type(torch.FloatTensor)
                        .to(self.device),
                    )
                    for j in range(self.num_classes)
                ]
            )
        )

        main_cost += self.adv_coeff * adv_cost

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
