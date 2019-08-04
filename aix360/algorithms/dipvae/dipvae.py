from __future__ import print_function

import torch
from torch import __init__
from torch.optim import Adam

from .dipvae_utils import VAE, DIPVAE
from .dipvae_utils import plot_reconstructions, plot_latent_traversal
from aix360.algorithms.die import DIExplainer

import os
import sys
import random
import time
import numpy as np


class DIPVAEExplainer(DIExplainer):
    """DIPVAEExplainer can be used to visualize the changes in the latent space of Disentangled Inferred Prior-VAE
    or DIPVAE [#1]_. This model is a Variational Autoencoder [#2]_ variant that leads to a
    disentangled latent space. This is achieved by matching the covariance of the prior distributions with the
    inferred prior.

    References:
        .. [#1] `Variational Inference of Disentangled Latent Concepts from Unlabeled Observations (DIP-VAE), ICLR 2018.
         Kumar, Sattigeri, Balakrishnan. <https://arxiv.org/abs/1711.00848>`_
        .. [#2] `Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. ICLR, 2014. <https://arxiv.org/pdf/1312.6114.pdf>`_

    """
    def __init__(self, model_args, dataset=None, net=None, cuda_available=None):
        """
        Initialize DIPVAEExplainer explainer.

        Args:
            model_args: This should contain all the parameter required for the generative model training and
                inference. This includes model type (vae, dipvae-i, dipvae-ii, user-defined). The user-defined model can be
                passed to the parameter net of the fit() function. Each of the model should have encoder and decode function
                defined. See the notebook example for other model specific parameters.
            dataset: The dataset object.
            net: If not None this is the user specified generative model.
            cuda_available: If True use GPU.
        """

        super(DIPVAEExplainer, self).__init__()
        self.model_args = model_args

        torch.manual_seed(self.model_args.seed)

        if net is None:
            if self.model_args.model == "vae":
                self.net = VAE(num_nodes=self.model_args.num_nodes, activation_type=self.model_args.activation_type,
                                 latent_dim=self.model_args.latent_dim,
                                 op_dim=np.prod(dataset.data_dims), args=self.model_args, cuda_available=cuda_available)
            elif self.model_args.model == "dipvae-i":
                self.net = DIPVAE(num_nodes=self.model_args.num_nodes, activation_type=self.model_args.activation_type,
                                    latent_dim=self.model_args.latent_dim,
                                    op_dim=np.prod(dataset.data_dims), args=self.model_args, cuda_available=cuda_available, mode='i',
                                    output_activation_type=dataset.output_activation_type, likelihood_type=dataset.likelihood_type)
            elif self.model_args.model == "dipvae-ii":
                self.net = DIPVAE(num_nodes=self.model_args.num_nodes, activation_type=self.model_args.activation_type,
                                    latent_dim=self.model_args.latent_dim,
                                    op_dim=np.prod(dataset.data_dims), args=self.model_args, cuda_available=cuda_available, mode='ii',
                                    output_activation_type=dataset.output_activation_type, likelihood_type=dataset.likelihood_type)
        else:
            self.net = net

        self.cuda_available = cuda_available
        self.dataset = dataset

        if self.cuda_available:
            self.net = self.net.cuda()

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        print("TBD: Implement set params in DIPVAEExplainer")

    def explain(self,
                input_images,
                edit_dim_id,
                edit_dim_value,
                edit_z_sample=False):

        """
        Edits the images in the latent space and returns the generated images.

        Args:
            input_images: The input images.
            edit_dim_id: The latent dimension id that need to be edited.
            edit_dim_value: The value that is assigned to the latent dimension with id edit_dim_id.
            edit_z_sample: If True will use the sample from encoder instead of the mean.
        Returns:
            Edited images.
        """

        reference_z, reference_mu, reference_std = self.net.encode(input_images)
        if edit_z_sample:
            edited_z = reference_z
            edited_z[:,edit_dim_id] = edit_dim_value
        else:
            edited_z = reference_mu
            edited_z[:, edit_dim_id] = edit_dim_value

        edited_images = self.net.decode(edited_z)

        return edited_images

    def fit(self, visualize=False, save_dir="results"):
        """
        Train the underlying generative model.

        Args:
            visualize: Plot reconstructions during fit.
            save_dir: directory where plots and model will be saved.
        Returns:
            elbo
        """

        torch.manual_seed(self.model_args.seed)

        optimizer = Adam(self.net.parameters(), lr=self.model_args.step_size)

        loss_epoch_list = []

        for epoch in np.arange(self.model_args.num_epochs):

            loss_epoch = 0.
            batch_id = 0

            for x, y in self.dataset.next_batch():
                #x, y = torch.tensor(x), torch.tensor(y)
                if self.cuda_available:
                    x = x.cuda()
                    y = y.cuda()

                # forward
                if "mnist" in self.dataset.name:
                    loss = self.net.neg_elbo(x.squeeze().view(-1, np.prod(self.dataset.data_dims)))
                else:
                    loss = self.net.neg_elbo(x)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch += loss

                batch_id += 1

                if visualize and batch_id % 10 == 0:


                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)

                    plot_reconstructions(self.dataset, self.net, x, image_id_to_plot=2, epoch=epoch,
                                         batch_id = batch_id, save_dir=save_dir)
                    if batch_id % 100 == 0:
                        plot_latent_traversal(self, x, self.model_args, self.dataset, image_id_to_plot=2, epoch=epoch,
                                              batch_id = batch_id, save_dir=save_dir)
                        torch.save(self.net, os.path.join(save_dir, 'net.p'))

            loss_epoch_list.append(-loss_epoch / self.dataset.num_training_instances)
            print("Epoch {0} | ELBO {1}".format(epoch, -loss_epoch / self.dataset.num_training_instances))



        return loss_epoch_list
