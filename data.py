import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import torch.distributions as dist
import abc
import torchvision.transforms as tr

class Data:
    def __init__(self, dataset, augmentations=None):
        self.dataset = dataset
        self.transforms = tr.Compose([tr.ToTensor()])

        if dataset == "MNIST":
            self.trainset = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=self.transforms)
            self.testset = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=self.transforms)


        x_train = self.trainset.data.reshape(-1, 28*28)/255.
        y_train = self.trainset.targets
        self.train_data = torch.utils.data.TensorDataset(x_train, y_train)

        x_test = self.testset.data.reshape(-1, 28*28)/255.
        y_test = self.testset.targets
        self.test_data = torch.utils.data.TensorDataset(x_test, y_test)

        if augmentations is not None:
            # increase the size of the training set by applying augmentations
            self.transforms_extra = tr.Compose([self.transforms, augmentations])
            self.trainset_extra = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=self.transforms_extra)

            x_train_extra = self.trainset.data.reshape(-1, 28*28)/255.
            y_train_extra = self.trainset.targets
            train_data_extra = torch.utils.data.TensorDataset(x_train_extra, y_train_extra)

            self.train_data = torch.utils.data.ConcatDataset([self.train_data, train_data_extra])


    def get_data(self):
        return self.train_data, self.test_data


    



    



    