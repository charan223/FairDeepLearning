#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.model_mlp import MLP

"""Module for LAFTR network
Parameters
----------
args: ArgumentParser
        Contains all model and shared input arguments
"""
class LaftrNet(nn.Module):
    def __init__(self, args):
        """Initializes LAFTR network: MLP encoder, MLP classifier, MLP discriminator"""
        super(LaftrNet, self).__init__()

        self.enc_neurons = [args.input_dim] + args.edepth * [args.ewidths] + [args.zdim]
        self.class_neurons = (
            [args.zdim] + args.cdepth * [args.cwidths] + [args.num_classes - 1]
        )
        if args.arch != "laftr-dp":
            self.adv_neurons = (
                [args.zdim + args.num_classes - 1]
                + args.adepth * [args.awidths]
                + [args.num_groups - 1]
            )
        else:
            self.adv_neurons = (
                [args.zdim] + args.adepth * [args.awidths] + [args.num_groups - 1]
            )

        self.encoder = MLP(self.enc_neurons)
        self.classifier = MLP(self.class_neurons)
        self.discriminator = MLP(self.adv_neurons)

        self.device = args.device
        self.arch = args.arch
        self.fair_coeff = args.fair_coeff
        self.class_coeff = 1.0
        self.aud_steps = args.aud_steps

        self.optimizer, self.optimizer_disc = self.get_optimizer()

    @staticmethod
    def build_model(args):
        """ Builds LAFTR network """
        model = LaftrNet(args)
        return model

    def get_optimizer(self):
        """Returns an optimizer for each network"""
        optimizer = torch.optim.Adam(self.predict_params())
        optimizer_disc = torch.optim.Adam(self.audit_params())
        return optimizer, optimizer_disc

    def predict_params(self):
        """Returns encoder and classifier parameters"""
        return list(self.classifier.parameters()) + list(self.encoder.parameters())

    def audit_params(self):
        """Returns discriminator parameters"""
        return self.discriminator.parameters()

    def l1_loss(self, y, y_logits):
        """Returns l1 loss"""
        y_hat = torch.sigmoid(y_logits)
        return torch.squeeze(torch.abs(y - y_hat))

    def ce_loss(self, y, y_logits, eps=1e-8):
        """Returns cross entropy loss"""
        y_hat = torch.sigmoid(y_logits)
        return -torch.sum(
            y * torch.log(y_hat + eps) + (1 - y) * torch.log(1 - y_hat + eps)
        )

    def get_weighted_aud_loss(self, L, X, Y, A, A_wts, Y_wts, AY_wts):
        """Returns weighted discriminator loss"""
        if self.arch == "laftr-dp":
            A0_wt = A_wts[0]
            A1_wt = A_wts[1]
            wts = A0_wt * (1 - A) + A1_wt * A
            wtd_L = L * torch.squeeze(wts)
        elif (
            self.arch == "laftr-eqodd"
            or self.arch == "laftr-eqopp0"
            or self.arch == "laftr-eqopp1"
        ):
            A0_Y0_wt = AY_wts[0][0]
            A0_Y1_wt = AY_wts[0][1]
            A1_Y0_wt = AY_wts[1][0]
            A1_Y1_wt = AY_wts[1][1]

            if self.arch == "laftr-eqodd":
                wts = (
                    A0_Y0_wt * (1 - A) * (1 - Y)
                    + A0_Y1_wt * (1 - A) * (Y)
                    + A1_Y0_wt * (A) * (1 - Y)
                    + A1_Y1_wt * (A) * (Y)
                )
            elif self.arch == "laftr-eqopp0":
                wts = A0_Y0_wt * (1 - A) * (1 - Y) + A1_Y0_wt * (A) * (1 - Y)
            elif self.arch == "laftr-eqopp1":
                wts = A0_Y1_wt * (1 - A) * (Y) + A1_Y1_wt * (A) * (Y)

            wtd_L = L * torch.squeeze(wts)
        else:
            raise Exception("Wrong model name")
            exit(0)

        return wtd_L

    def forward(self, X, Y, A, mode="train", A_wts=None, Y_wts=None, AY_wts=None, reweight_target_tensor=None, reweight_attr_tensors=None):
        """Computes forward pass through encoder ,
            Computes backward pass on the target function"""
        Z = self.encoder(X)
        Y_logits = torch.squeeze(self.classifier(Z))

        if self.arch != "laftr-dp":
            Z = torch.cat(
                [Z, torch.unsqueeze(Y.type(torch.FloatTensor), 1).to(self.device)],
                axis=1,
            )
        class_loss = self.class_coeff * torch.mean(self.ce_loss(Y, Y_logits))

        # discriminator loss
        A_logits = torch.squeeze(self.discriminator(Z))
        aud_loss = -self.fair_coeff * self.l1_loss(A, A_logits)
        weighted_aud_loss = self.get_weighted_aud_loss(
            aud_loss, X, Y, A, A_wts, Y_wts, AY_wts
        )
        weighted_aud_loss = torch.mean(weighted_aud_loss)

        loss = class_loss + weighted_aud_loss

        torch.autograd.set_detect_anomaly(True)

        # train (encoder, classifier), discriminator alternatively
        if mode == "train":
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.predict_params(), 5.0)
            self.optimizer.step()

            for i in range(self.aud_steps):
                self.optimizer_disc.zero_grad()
                if i != self.aud_steps - 1:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.audit_params(), 5.0)
                self.optimizer_disc.step()

        cost_dict = dict(
            main_cost=loss,
        )

        logits = torch.sigmoid(Y_logits)

        return logits, cost_dict
