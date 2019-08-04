import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear

import os

td = torch.distributions


def convert_and_reshape(x, dataset_obj):
    return x.cpu().data.numpy().reshape([-1] + dataset_obj.data_dims).transpose(0,2,3,1)


def plot_reconstructions(dataset_obj, trained_net, input_images, image_id_to_plot=0, epoch=0, batch_id = 0, save_dir="results"):
    z_sample, _, _ = trained_net.encode(input_images)
    #print(z_sample[image_id_to_plot])
    recons = trained_net.decode(z_sample)

    input_images_numpy = convert_and_reshape(input_images, dataset_obj)
    recons_numpy = convert_and_reshape(recons, dataset_obj)

    f, axarr = plt.subplots(1, 2)
    if input_images_numpy.shape[2] == 1:
        axarr[0].imshow(input_images_numpy[image_id_to_plot, :, :, 0], cmap="gray")
        axarr[1].imshow(recons_numpy[image_id_to_plot, :, :, 0], cmap="gray")
    else:
        axarr[0].imshow(input_images_numpy[image_id_to_plot]*0.5 + 0.5)
        axarr[1].imshow(recons_numpy[image_id_to_plot]*0.5 + 0.5)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    f.savefig(os.path.join(save_dir, 'recons_epoch_{}_batch_id_{}.png'.format(epoch, batch_id)))
    plt.close(fig=f)


def plot_latent_traversal(explainer, input_images, args, dataset_obj, image_id_to_plot=0, num_sweeps=15,
                          max_abs_edit_value=10.0, epoch=0, batch_id = 0, save_dir="results"):
    edit_dim_values = np.linspace(-1.0 *max_abs_edit_value, max_abs_edit_value, num_sweeps)

    f, axarr = plt.subplots(args.latent_dim, len(edit_dim_values), sharex=True, sharey=True)
    f.set_size_inches(10, 10* args.latent_dim / len(edit_dim_values))

    for i in range(args.latent_dim):
        for j in range(len(edit_dim_values)):

            edited_images = convert_and_reshape(explainer.explain(input_images=input_images,
                             edit_dim_id = i,
                             edit_dim_value = edit_dim_values[j],edit_z_sample=False), dataset_obj)
            if edited_images.shape[2] == 1:
                axarr[i][j].imshow(edited_images[image_id_to_plot,:,:,0], cmap="gray", aspect='auto')
            else:
                axarr[i][j].imshow(edited_images[image_id_to_plot]*0.5 + 0.5, aspect='auto')
            #axarr[j][i].axis('off')
            if i == len(axarr) - 1:
                axarr[i][j].set_xlabel("z:" + str(np.round(edit_dim_values[j], 1)))
            if j == 0:
                axarr[i][j].set_ylabel("l:" + str(i))
            axarr[i][j].set_yticks([])
            axarr[i][j].set_xticks([])
    plt.subplots_adjust(hspace=0, wspace=0)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    f.savefig(os.path.join(save_dir, 'traversal_epoch_{}_batch_id_{}.png'.format(epoch, batch_id)))
    plt.close(fig=f)


def reparam(mu, std, do_sample=True, cuda=True):
    """Reparametrization for Normal distribution.
    """
    if do_sample:
        eps = torch.FloatTensor(std.size()).normal_()
        if cuda:
            eps = eps.cuda()
        eps = Variable(eps)
        return mu + eps * std
    else:
        return mu


def bernoulli_likelihood(x, x_pred, dim=None):
    """Compute Bernoulli likelihood.
    """
    assert x_pred.min() >= 0.
    assert x_pred.max() <= 1.
    lik = td.Bernoulli(probs=x_pred).log_prob(x)
    if torch.isnan(lik.sum()):
        print (lik.sum())
    if dim:
        return lik.sum(dim=dim)
    else:
        return lik.sum()


def gaussian_likelihood(x, x_pred):
    """Compute Gaussian likelihood.
    """
    return - torch.sum(F.mse_loss(x_pred, x, reduction='none'))


class FCNet(nn.Module):
    def __init__(self, num_nodes=50, ip_dim=1, op_dim=1, activation_type='relu', args=None):
        super(FCNet, self).__init__()
        self.args = args
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            print("Activation Type not supported")
            return
        layer = Linear
        self.fc_hidden = []
        self.fc1 = layer(ip_dim, num_nodes)
        self.bn1 = nn.BatchNorm1d(num_nodes)
        for _ in np.arange(self.args.num_layers - 1):
            self.fc_hidden.append(layer(num_nodes, num_nodes))
            self.fc_hidden.append(nn.BatchNorm1d(num_nodes))
            self.fc_hidden.append(self.activation)
        self.features = nn.Sequential(*self.fc_hidden)
        self.fc_out = layer(num_nodes, op_dim)

    def forward(self, x):
        x = x.squeeze().view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.features(x)
        return self.fc_out(x)


class VAE(nn.Module):
    """
    Variational Autoencoder [1]_ is an generative model with stochastic encoder and decoder learned using Bayesian
    variational inference. The encoder network is used to obtain latent representations for the input while the decoder
    generated the samples from the latents.

    References:
        .. [1] Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. ICLR, 2014
    """
    def __init__(self, num_nodes=50, latent_dim=10, op_dim=784, activation_type='relu',
                 args=None, cuda_available=None, init_networks=True):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        if init_networks:
            args.num_layers = args.num_gen_layers
            generative_op_dim = op_dim
            self.generative_net = FCNet(num_nodes=num_nodes, op_dim=generative_op_dim, ip_dim=latent_dim,
                                        activation_type=activation_type, args=args)
            args.num_layers = args.num_inference_layers
            self.inference_net = FCNet(num_nodes=num_nodes, op_dim=2 * latent_dim, ip_dim=op_dim,
                                       activation_type=activation_type,
                                       args=args)
        self.z_prior_stdv = torch.Tensor([1.])
        self.z_prior_mean = torch.Tensor([0.])
        if cuda_available:
            self.z_prior_stdv = self.z_prior_stdv.cuda()
            self.z_prior_mean = self.z_prior_mean.cuda()
        self.cuda_available = cuda_available
        self.name = "VAE"
        self.args = args

    def encode(self, x):
        z = self.inference_net.forward(x)
        mu_z = z[:, :self.latent_dim]
        std_z = F.softplus(z[:, self.latent_dim:])
        z_sample = reparam(mu_z, std_z, cuda=self.cuda_available)
        return z_sample, mu_z, std_z

    def decode(self, z):
        return torch.sigmoid(self.generative_net(z))

    def forward(self, x, return_z_sample=False):
        z_sample, mu_z, std_z = self.encode(x)
        if return_z_sample:
            return self.generative_net(z_sample), mu_z, std_z, z_sample
        else:
            return self.generative_net(z_sample), mu_z, std_z

    def sample_from_gen_model(self, num_samples=25):
        self.eval()
        z_samples = torch.randn(num_samples, self.latent_dim)
        if self.cuda_available:
            z_samples = z_samples.cuda()
        self.train()
        return torch.sigmoid(self.generative_net(z_samples))

    def _kl_divergence_z(self, mu_z, std_z):
        """
        """
        kld_z = self.z_prior_stdv.log() - std_z.log() + (std_z ** 2 + (mu_z.pow(2) - self.z_prior_mean)) /\
                (2 * self.z_prior_stdv.pow(2)) - 0.5
        return kld_z.sum()

    def neg_elbo(self, x, beta=None):
        recon, mu_z, std_z = self.forward(x)
        recon = torch.sigmoid(recon)
        kl_z = self._kl_divergence_z(mu_z, std_z)
        if beta is not None:
            kl_z = beta * kl_z
        return - bernoulli_likelihood(x, recon) + kl_z


class DIPVAE(nn.Module):
    """
    Disentangled Inferred Prior-VAE or DIPVAE [1] is a Variational Autoencoder [2]_ variant that leads to a
    disentangled latent space. This is achieved by matching the covariance of the prior distributions with the
    inferred prior.

    References:
        .. [1] Variational Inference of Disentangled Latent Concepts from Unlabeled Observations (DIP-VAE), ICLR 2018.
         Kumar, Sattigeri, Balakrishnan. https://arxiv.org/abs/1711.00848
        .. [2] Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. ICLR, 2014
    """
    def __init__(self, num_nodes=50, latent_dim=10, op_dim=784, activation_type='relu',
                 args=None, cuda_available=None, init_networks=True, mode=None, output_activation_type=None,
                 likelihood_type=None, beta=1.0):
        super(DIPVAE, self).__init__()
        self.latent_dim = latent_dim
        self.mode = mode

        if init_networks:
            args.num_layers = args.num_gen_layers
            generative_op_dim = op_dim
            self.generative_net = FCNet(num_nodes=num_nodes, op_dim=generative_op_dim, ip_dim=latent_dim,
                                        activation_type=activation_type, args=args)
            args.num_layers = args.num_inference_layers
            self.inference_net = FCNet(num_nodes=num_nodes, op_dim=2 * latent_dim, ip_dim=op_dim,
                                       activation_type=activation_type,
                                       args=args)
        self.z_prior_stdv = torch.Tensor([1.])
        self.z_prior_mean = torch.Tensor([0.])
        if cuda_available:
            self.z_prior_stdv = self.z_prior_stdv.cuda()
            self.z_prior_mean = self.z_prior_mean.cuda()
        self.cuda_available = cuda_available
        self.name = "DIPVAE"
        self.args = args

        self.beta = beta

        self.lambda_diag_factor = args.lambda_diag_factor
        self.lambda_offdiag = args.lambda_offdiag
        self.lambda_diag = self.lambda_diag_factor * self.lambda_offdiag

        self.output_activation_type = output_activation_type
        if likelihood_type == "bernoulli":
            self.likelihood = bernoulli_likelihood
        elif likelihood_type == "gaussian":
            self.likelihood = gaussian_likelihood
        else:
            raise NotImplementedError("Unsupported likelihood type.")

    def encode(self, x):
        z = self.inference_net.forward(x)
        mu_z = z[:, :self.latent_dim]
        std_z = F.softplus(z[:, self.latent_dim:])
        z_sample = reparam(mu_z, std_z, cuda=self.cuda_available)
        return z_sample, mu_z, std_z

    def decode(self, z):
        if self.output_activation_type is None:
            return self.generative_net(z)
        elif self.output_activation_type == "sigmoid":
            return torch.sigmoid(self.generative_net(z))
        elif self.output_activation_type == "tanh":
            return torch.tanh(self.generative_net(z))
        else:
            raise NotImplementedError("Unsupported output activation type.")

    def forward(self, x, return_z_sample=False):
        z_sample, mu_z, std_z = self.encode(x)
        if return_z_sample:
            return self.decode(z_sample), mu_z, std_z, z_sample
        else:
            return self.decode(z_sample), mu_z, std_z

    def sample_from_gen_model(self, num_samples=25):
        self.eval()
        z_samples = torch.randn(num_samples, self.latent_dim)
        if self.cuda_available:
            z_samples = z_samples.cuda()
        self.train()
        return self.decode(z_samples)

    def _get_covariance_mu_z(self, mu_z):

        zero_mean_mu_z = mu_z - torch.mean(mu_z, dim=1, keepdim=True)
        zero_mean_mu_z_t = zero_mean_mu_z.t()

        return zero_mean_mu_z_t.matmul(zero_mean_mu_z).squeeze()

    def _get_dipvae_regularizer(self, cov_z, lambda_offdiag, lambda_diag):
        cov_z_diag = torch.diag(cov_z)
        cov_z_offdiag = cov_z - torch.diag(cov_z_diag)

        return lambda_offdiag * torch.sum(cov_z_offdiag ** 2) + lambda_diag * torch.sum((cov_z_diag - 1) ** 2)

    def _regularizer(self, mu_z, std_z):
        kld_z = self.z_prior_stdv.log() - std_z.log() + (std_z ** 2 + (mu_z.pow(2) - self.z_prior_mean)) /\
                (2 * self.z_prior_stdv.pow(2)) - 0.5
        regularizer_loss = kld_z.sum()

        regularizer_loss = self.beta * regularizer_loss

        cov_mu_z = self._get_covariance_mu_z(mu_z)

        if self.mode == "i":
            dipvae_regularizer_loss = self._get_dipvae_regularizer(cov_mu_z, self.lambda_offdiag, self.lambda_diag)

        elif self.mode == "ii":
            cov_z = cov_mu_z + torch.mean(torch.diag(std_z**2), dim=0)
            dipvae_regularizer_loss = self._get_dipvae_regularizer(cov_z, self.lambda_offdiag, self.lambda_diag)
        else:
            raise NotImplementedError("Unsupported dipvae mode.")

        return regularizer_loss + dipvae_regularizer_loss

    def neg_elbo(self, x):
        recon, mu_z, std_z = self.forward(x)
        regularizer = self._regularizer(mu_z, std_z)
        return - self.likelihood(x, recon) + regularizer
