#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


"""Module for a VAE Convolutional Neural Network for CI-MNIST dataset
Parameters
----------
args: ArgumentParser
        Contains all model and shared input arguments
"""
class ConvVAE(nn.Module):
    # Input to ConvVAE is resized to 32 * 32 * 3, each pixel has 3 float values
    # between 0, 1 to represent each of RGB channels
    def __init__(self, args):
        """Initialized VAE CNN"""
        super(ConvVAE, self).__init__()

        self.z_dim = args.zdim
        self.batch_size = args.batch_size

        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(True),
            Resize((-1, 1024)),
            nn.Linear(1024, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 2 * self.z_dim)
        )

        self.dec = nn.Sequential(
            nn.Linear(self.z_dim, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 1024),
            nn.LeakyReLU(True),
            Resize((-1, 64, 4, 4)),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode="normal"):
        """Initializes weights of VAE parameters"""
        for block in self._modules:
            for m in self._modules[block]:
                if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                    torch.nn.init.xavier_uniform_(m.weight)

    def encode(self, x):
        """encodes input, returns statistics of the distribution"""
        stats = self.enc(x).squeeze()
        mu, logvar = stats[:, : self.z_dim], stats[:, self.z_dim :]
        return mu, logvar

    def decode(self, z):
        """decodes latent representation"""
        stats = self.dec(z)
        return stats


"""Module for a VAE MLP for Adult dataset
Parameters
----------
num_neurons: list, dtype: int
        Number of neurons in each layer
zdim: int
        length of mu, std of the latent representation
activ: string, default: "leakyrelu"
        Activation function
"""
class MLP(nn.Module):
    def __init__(self, num_neurons, zdim, activ="leakyrelu"):
        """Initialized VAE MLP"""
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
        self.zdim = zdim

    def forward(self, inputs, mode):
        """Computes forward pass for VAE's encoder, VAE's decoder, classifier, discriminator"""
        L = inputs
        for hidden in self.hiddens:
            L = F.leaky_relu(hidden(L))
        if mode == "encode":
            mu, logvar = L[:, : self.zdim], L[:, self.zdim :]
            return mu, logvar
        elif mode == "decode":
            return L.squeeze()
        elif mode == "discriminator":
            logits, probs = L, nn.Softmax(dim=1)(L)
            return logits, probs
        elif mode == "classify":
            return L
        else:
            raise Exception(
                "Wrong mode choose one of encoder/decoder/discriminator/classifier"
            )
        return

"""Module for FFVAE network
Parameters
----------
args: ArgumentParser
        Contains all model and shared input arguments
"""
class Ffvae(nn.Module):
    """Initializes FFVAE network: VAE encoder, MLP classifier, MLP discriminator"""
    def __init__(self, args):
        super(Ffvae, self).__init__()
        self.input_dim = args.input_dim
        self.num_classes = args.num_classes
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.zdim = args.zdim
        self.sensattr = args.sensattr
        self.device = args.device
        self.data = args.data

        # VAE encoder
        if self.data == "clr-mnist":
            self.conv_vae = ConvVAE(args)
        else:
            self.enc_neurons = (
                [self.input_dim] + args.edepth * [args.ewidths] + [2 * args.zdim]
            )
            self.encoder = MLP(self.enc_neurons, self.zdim).to(args.device)

            self.dec_neurons = (
                [args.zdim] + args.edepth * [args.ewidths] + [args.input_dim]
            )
            self.decoder = MLP(self.dec_neurons, self.zdim).to(args.device)

        # MLP Discriminator
        self.adv_neurons = [args.zdim] + args.adepth * [args.awidths] + [2]
        self.discriminator = MLP(self.adv_neurons, args.zdim).to(args.device)

        # MLP Classifier
        self.class_neurons = (
            [args.zdim] + args.cdepth * [args.cwidths] + [args.num_classes]
        )
        self.classifier = MLP(self.class_neurons, args.zdim).to(args.device)

        # index for sensitive attribute
        self.n_sens = 1
        self.sens_idx = list(range(self.n_sens))
        self.nonsens_idx = [
            i for i in range(int(self.zdim / 2)) if i not in self.sens_idx
        ]
        self.count = 0
        self.batch_size = args.batch_size

        (
            self.optimizer_ffvae,
            self.optimizer_disc,
            self.optimizer_class,
        ) = self.get_optimizer()

    @staticmethod
    def build_model(args):
        """ Builds FFVAE class """
        model = Ffvae(args)
        return model

    def vae_params(self):
        """Returns VAE parameters required for training VAE"""
        if self.data == "clr-mnist":
            return list(self.conv_vae.parameters())
        else:
            return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def discriminator_params(self):
        """Returns discriminator parameters"""
        return list(self.discriminator.parameters())

    def classifier_params(self):
        """Returns classifier parameters"""
        return list(self.classifier.parameters())

    def get_optimizer(self):
        """Returns an optimizer for each network"""
        optimizer_ffvae = torch.optim.Adam(self.vae_params())
        optimizer_disc = torch.optim.Adam(self.discriminator_params())
        optimizer_class = torch.optim.Adam(self.classifier_params())
        return optimizer_ffvae, optimizer_disc, optimizer_class

    def forward(self, inputs, labels, attrs, mode="train", A_wts=None, Y_wts=None, AY_wts=None, reweight_target_tensor=None, reweight_attr_tensors=None):
        """Computes forward pass through encoder ,
            Computes backward pass on the target function"""
        # Make inputs between 0, 1
        x = (inputs + 1) / 2

        # encode: get q(z,b|x)
        if self.data == "clr-mnist":
            _mu, _logvar = self.conv_vae.encode(x)
        else:
            _mu, _logvar = self.encoder(x, "encode")

        # only non-sensitive dims of latent code modeled as Gaussian
        mu = _mu[:, self.nonsens_idx]
        logvar = _logvar[:, self.nonsens_idx]
        zb = torch.zeros_like(_mu) 
        std = (logvar / 2).exp()
        q_zIx = torch.distributions.Normal(mu, std)

        # the rest are 'b', deterministically modeled as logits of sens attrs a
        b_logits = _mu[:, self.sens_idx]

        # draw reparameterized sample and fill in the code
        z = q_zIx.rsample()
        # reparametrization
        zb[:, self.sens_idx] = b_logits
        zb[:, self.nonsens_idx] = z

        # decode: get p(x|z,b)
        # xIz_params = self.decoder(zb, "decode")  # decoder yields distn params not preds
        if self.data == "clr-mnist":
            xIz_params = self.conv_vae.decode(zb)
            xIz_params = torch.sigmoid(xIz_params)
        else:
            xIz_params = self.decoder(zb, "decode")

        p_xIz = torch.distributions.Normal(loc=xIz_params, scale=1.0)
        # negative recon error per example
        logp_xIz = p_xIz.log_prob(x)  

        recon_term = logp_xIz.reshape(len(x), -1).sum(1)

        # prior: get p(z)
        p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        # compute analytic KL from q(z|x) to p(z), then ELBO
        kl = torch.distributions.kl_divergence(q_zIx, p_z).sum(1)

        # vector el
        elbo = recon_term - kl  

        # decode: get p(a|b)
        clf_losses = [
            nn.BCEWithLogitsLoss()(_b_logit.to(self.device), _a_sens.to(self.device))
            for _b_logit, _a_sens in zip(
                b_logits.squeeze().t(), attrs.type(torch.FloatTensor).t()
            )
        ]

        # compute loss
        logits_joint, probs_joint = self.discriminator(zb, "discriminator")
        total_corr = logits_joint[:, 0] - logits_joint[:, 1]

        ffvae_loss = (
            -1.0 * elbo.mean()
            + self.gamma * total_corr.mean()
            + self.alpha * torch.stack(clf_losses).mean()
        )

        # shuffling minibatch indexes of b0, b1, z
        z_fake = torch.zeros_like(zb)
        z_fake[:, 0] = zb[:, 0][torch.randperm(zb.shape[0])]
        z_fake[:, 1:] = zb[:, 1:][torch.randperm(zb.shape[0])]
        z_fake = z_fake.to(self.device).detach()

        # discriminator
        logits_joint_prime, probs_joint_prime = self.discriminator(
            z_fake, "discriminator"
        )
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        disc_loss = (
            0.5
            * (
                F.cross_entropy(logits_joint, zeros)
                + F.cross_entropy(logits_joint_prime, ones)
            ).mean()
        )

        encoded_x = _mu.detach()

        # IMPORTANT: randomizing sensitive latent
        encoded_x[:, 0] = torch.randn_like(encoded_x[:, 0])

        pre_softmax = self.classifier(encoded_x, "classify")
        logprobs = F.log_softmax(pre_softmax, dim=1)
        class_loss = F.nll_loss(logprobs, labels)

        cost_dict = dict(
            ffvae_cost=ffvae_loss, disc_cost=disc_loss, main_cost=class_loss
        )

        # ffvae optimization
        if mode == "ffvae_train":
            self.optimizer_ffvae.zero_grad()
            ffvae_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.vae_params(), 5.0)
            self.optimizer_ffvae.step()

            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator_params(), 5.0)
            self.optimizer_disc.step()

        # classifier optimization
        elif mode == "train":
            self.optimizer_class.zero_grad()
            class_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier_params(), 5.0)
            self.optimizer_class.step()

        return pre_softmax, cost_dict

"""Function to resize input
Parameters
----------
size: tuple
        target size to be resized to
tensor: torch tensor
        input tensor
Returns
----------
Resized tensor
"""
class Resize(torch.nn.Module):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
