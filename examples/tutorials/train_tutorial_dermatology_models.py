import torch
from torchvision import transforms
import torch.nn as nn

import pandas as pd
from PIL import Image
import numpy as np
import math
import os

import matplotlib.pyplot as plt

from aix360.algorithms.dipvae.dipvae_utils import VAE, DIPVAE
from aix360.algorithms.dipvae import DIPVAEExplainer
from aix360.algorithms.dipvae.dipvae_utils import plot_reconstructions, plot_latent_traversal


class ISICImages(torch.utils.data.Dataset):
    def __init__(self, root, file_path, transform=None):
        self.root = root
        self.data  = pd.read_csv(file_path)

        self.data['disease_class'] = self.data.apply(lambda x: self.data.columns[x==1], axis = 1)
        self.data['disease_class'] = self.data['disease_class'].apply(lambda x: str(x[0]))

        diseases = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
        self.data['disease_class_int'] = self.data['disease_class'].map(diseases)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.root + self.data.iloc[index, 0]+'.jpg'
        img = Image.open(img_path).convert("RGB")
        #imghsv= Image.open(img_path).convert('HSV')

        label = self.data.iloc[index, 9]

        if self.transform is not None:
            img = self.transform(img)
            #imghsv = self.transform(imghsv)

        #img_6channel = np.vstack((img, imghsv))

        return img, label #img_6channel


class ISICDataset():
    def __init__(self, batch_size=256,
                 root_images_path='./data/ISIC2018_Task3_Training_Input/',
                 file_path_labels='./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'):

        img_size = (128, 128)  # (224,224)

        # Transforms
        transform = transforms.Compose([
            transforms.Resize(size=img_size),
            transforms.ToTensor()

        ])

        train_set = ISICImages(root=root_images_path, file_path=file_path_labels, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True)

        self.name = "isic"
        self.data_dims = [3, 128, 128]
        self.train_size = len(self.train_loader)
        self.range = [-1.0, 1.0]
        self.batch_size = batch_size
        self.num_training_instances = len(train_set)
        self.likelihood_type = "gaussian"
        self.output_activation_type = "tanh"

    def next_batch(self):
        for x, y in self.train_loader:
            x = 2.0 * (x - 0.5)
            yield x, y


class ConvEncNet(nn.Module):
    def __init__(self, latent_dim=20, num_filters=64, num_channels=3, image_size=128, activation_type='relu', args=None):
        super(ConvEncNet, self).__init__()
        self.args = args
        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'tanh':
            self.activation = nn.Tanh(inplace=True)
        else:
            print("Activation Type not supported")
            return

        self.conv_hidden = []
        self.conv1 = nn.Conv2d(num_channels, num_filters, 4, 2, 1, bias=True)

        num_layers = math.log2(image_size)
        assert num_layers == round(num_layers), 'Image size that are power of 2 are supported.'
        num_layers = int(num_layers)

        for i in np.arange(num_layers - 3):
            self.conv_hidden.append(nn.Conv2d(num_filters * 2 ** i, num_filters * 2 ** (i + 1), 4, 2, 1, bias=True))
            self.conv_hidden.append(nn.BatchNorm2d(num_filters * 2 ** (i + 1)))
            self.conv_hidden.append(self.activation)

        self.features = nn.Sequential(*self.conv_hidden)
        self.conv_mu = nn.Conv2d(num_filters * 2 ** (num_layers - 3), latent_dim, 4)
        self.conv_var = nn.Conv2d(num_filters * 2 ** (num_layers - 3), latent_dim, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.features(x)
        mu_z = self.conv_mu(x).view(x.size(0), -1)
        var_z = self.conv_var(x).view(x.size(0), -1)
        return torch.cat([mu_z, var_z], dim=1)


class ConvDecNet(nn.Module):
    def __init__(self, latent_dim=20, num_filters=64, num_channels=3, image_size=128, activation_type='relu', args=None):
        super(ConvDecNet, self).__init__()
        self.args = args
        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'tanh':
            self.activation = nn.Tanh(inplace=True)
        else:
            print("Activation Type not supported")
            return

        self.deconv_hidden = []

        num_layers = math.log2(image_size)
        assert num_layers == round(num_layers), 'Image size that are power of 2 are supported.'
        num_layers = int(num_layers)

        self.deconv1 = nn.ConvTranspose2d(latent_dim, num_filters * 2 ** (num_layers - 3), 4, 1, 0, bias=True)
        self.bn1 = nn.BatchNorm2d(num_filters * 2 ** (num_layers - 3))

        for i in np.arange(num_layers - 3, 0, -1):
            self.deconv_hidden.append(nn.ConvTranspose2d(num_filters * 2 ** i,
                                                         num_filters * 2 ** (i - 1),
                                                         4, 2, 1, bias=True))
            self.deconv_hidden.append(nn.BatchNorm2d(num_filters * 2 ** (i - 1)))
            self.deconv_hidden.append(self.activation)

        self.features = nn.Sequential(*self.deconv_hidden)

        self.deconv_out = nn.ConvTranspose2d(num_filters, num_channels, 4, 2,1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.features(x)
        return self.deconv_out(x)


class ConvDIPVAE(DIPVAE):
    """
    """
    def __init__(self, num_filters=50, latent_dim=10, num_channels=3, image_size=128, activation_type='relu',
                 args=None, cuda_available=None, init_networks=True, mode=None, output_activation_type='tanh',
                 likelihood_type=None, beta=1.0):
        super(ConvDIPVAE, self).__init__(latent_dim=latent_dim,
                                     activation_type=activation_type,
                                     args=args, cuda_available=cuda_available, init_networks=False, mode=mode,
                                     output_activation_type=output_activation_type,likelihood_type=likelihood_type,
                                         beta=beta)

        if init_networks:
            self.generative_net = ConvDecNet(latent_dim=latent_dim, num_filters=num_filters, num_channels=num_channels,
                                             image_size=image_size,activation_type=activation_type, args=args)
            self.inference_net = ConvEncNet(latent_dim=latent_dim, num_filters=num_filters, num_channels=num_channels,
                                            image_size=image_size, activation_type=activation_type, args=args)

        self.name = "ConvDIPVAE"


if __name__ == '__main__':

    import argparse

    cuda_available = torch.cuda.is_available()
    print("CUDA: {}".format(cuda_available))

    dataset_obj = ISICDataset(root_images_path='./data/Task3/ISIC2018_Task3_Training_Input/',
                              file_path_labels='./data/Task3/ISIC2018_Task3_Training_GroundTruth/'
                                              'ISIC2018_Task3_Training_GroundTruth.csv',
                              batch_size=32)

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='user-defined')
    parser.add_argument('--activation_type', type=str, default='relu')
    parser.add_argument('--num_filters', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--num_channels', type=int, default=dataset_obj.data_dims[0])
    parser.add_argument('--image_size', type=int, default=dataset_obj.data_dims[-1])
    parser.add_argument('--step_size', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lambda_diag_factor', type=float, default=0.)
    parser.add_argument('--lambda_offdiag', type=float, default=0.)
    parser.add_argument('--output_activation_type', type=str, default=dataset_obj.output_activation_type)
    parser.add_argument('--likelihood_type', type=str, default=dataset_obj.likelihood_type)
    parser.add_argument('--mode', type=str, default='ii')
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fit', type=int, default=1)
    parser.add_argument('--root_save_dir', type=str, default='.')

    model_args = parser.parse_args()

    setup = [
        ('model={:s}', model_args.model),
        ('lambda_diag_factor={:.0e}', model_args.lambda_diag_factor),
        ('lambda_offdiag={:.0e}', model_args.lambda_offdiag),
        ('beta={:.0e}', model_args.beta),
    ]
    save_dir = os.path.join(model_args.root_save_dir, "results"+'_'.join([t.format(v) for (t, v) in setup]))

    if model_args.fit:

        net = ConvDIPVAE(num_filters=model_args.num_filters, latent_dim=model_args.latent_dim,
                     num_channels=model_args.num_channels, image_size=model_args.image_size,
                     activation_type=model_args.activation_type, args=model_args, cuda_available=cuda_available,
                     init_networks=True, mode=model_args.mode, output_activation_type=model_args.output_activation_type,
                     likelihood_type=model_args.likelihood_type, beta=model_args.beta)
    else:
        net = torch.load(os.path.join(save_dir, 'net.p'))

    dipvaeii_explainer = DIPVAEExplainer(net=net, dataset=dataset_obj, cuda_available=cuda_available,
                                         model_args=model_args)

    if model_args.fit:
        loss_epoch_list = dipvaeii_explainer.fit(visualize=True,
                                                 save_dir=save_dir)

    # After training
    for x, _ in dataset_obj.next_batch():
        if dipvaeii_explainer.cuda_available:
            x = x.cuda()
        plot_reconstructions(dipvaeii_explainer.dataset, dipvaeii_explainer.net, x, image_id_to_plot=2,
                             epoch='end', batch_id = 'end', save_dir=save_dir)
        plot_latent_traversal(dipvaeii_explainer, x, dipvaeii_explainer.model_args, dipvaeii_explainer.dataset,
                              image_id_to_plot=2, epoch='end', batch_id='end', save_dir=save_dir)
        break
