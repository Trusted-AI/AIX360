import unittest

from aix360.algorithms.dipvae import DIPVAEExplainer

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse
import numpy as np


class FakeFMnistDataset():
    def __init__(self, batch_size=256, subset_size=256, test_batch_size=256):
        trans = transforms.Compose([transforms.ToTensor()])

        root = './data_fake_fmnist'
        train_set = dset.FakeData(image_size=(1, 28, 28),transform=transforms.ToTensor())
        test_set = dset.FakeData(image_size=(1, 28, 28),transform=transforms.ToTensor())

        indices = torch.randperm(len(train_set))[:subset_size]
        train_set = torch.utils.data.Subset(train_set, indices)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=test_batch_size,
            shuffle=False)

        self.name = "fakemnist"
        self.data_dims = [28, 28, 1]
        self.train_size = len(self.train_loader)
        self.test_size = len(self.test_loader)
        self.range = [0.0, 1.0]
        self.batch_size = batch_size
        self.num_training_instances = len(train_set)
        self.num_test_instances = len(test_set)
        self.likelihood_type = 'gaussian'
        self.output_activation_type = 'sigmoid'

    def next_batch(self):
        for x, y in self.train_loader:
            #print(x.size())
            #x = np.reshape(x, (-1, 28, 28, 1))
            yield x, None

    def next_test_batch(self):
        for x, y in self.test_loader:
            #x = np.reshape(x, (-1, 28, 28, 1))
            yield x, None


class TestDIPVAEExplainer(unittest.TestCase):

    def test_DIPVAEExplainer(self):

        # Load the dataset object
        dataset_obj = FakeFMnistDataset()
        self.assertIsNotNone(dataset_obj)

        cuda_available = torch.cuda.is_available()

        # Initialize model arguments
        dipvaeii_args = argparse.Namespace()

        dipvaeii_args.model = 'dipvae-ii'

        dipvaeii_args.activation_type = 'relu'
        dipvaeii_args.num_nodes = 1200
        dipvaeii_args.latent_dim = 10
        dipvaeii_args.num_gen_layers = 3
        dipvaeii_args.num_inference_layers = 2

        dipvaeii_args.step_size = 0.001
        dipvaeii_args.lambda_diag_factor = 10.0
        dipvaeii_args.lambda_offdiag = 0.001

        dipvaeii_args.seed = 0

        # Run for 2 epochs for testing elbo increase
        dipvaeii_args.num_epochs = 2

        # Initialize the explainer model
        dipvaeii_explainer = DIPVAEExplainer(model_args=dipvaeii_args, dataset=dataset_obj, cuda_available=cuda_available)

        # Fit the generative model.
        loss_epoch_list = dipvaeii_explainer.fit()

        for x, _ in dataset_obj.next_test_batch():
            input_images = x.squeeze().view(-1, np.prod(dataset_obj.data_dims))
            break

        convert_and_reshape = lambda x: x.cpu().data.numpy().reshape([-1] + dataset_obj.data_dims)
        edited_images = dipvaeii_explainer.explain(input_images=input_images,
                                                              edit_dim_id=0,
                                                              edit_dim_value=-1.0, edit_z_sample=True)

        # Make sure edited images are different
        self.assertGreater(np.linalg.norm(
            np.abs(convert_and_reshape(input_images) - convert_and_reshape(edited_images)))
            , 0.0)


if __name__ == '__main__':
    unittest.main()
