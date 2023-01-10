import os
import numpy as np
import torch
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
import torchvision
import torchvision.transforms as tr
from torchvision.datasets import MNIST, FashionMNIST

class Data:
    def __init__(self, dataset, augmentations=None):
        self.dataset = dataset
        self.transforms = tr.Compose([tr.ToTensor()])

        if dataset == "MNIST":
            self.trainset = MNIST(root="data", train=True, download=True, transform=self.transforms)
            self.testset = MNIST(root="data", train=False, download=True, transform=self.transforms)

        elif dataset == "FashionMNIST":
            self.trainset = FashionMNIST(root="data", train=True, download=True, transform=self.transforms)
            self.testset = FashionMNIST(root="data", train=False, download=True, transform=self.transforms)


        x_train = self.trainset.data.reshape(-1, 28*28)/255.
        y_train = self.trainset.targets
        self.train_data = TensorDataset(x_train, y_train)

        x_test = self.testset.data.reshape(-1, 28*28)/255.
        y_test = self.testset.targets
        self.test_data = TensorDataset(x_test, y_test)

        if augmentations is not None:
            # increase the size of the training set by applying augmentations
            self.transforms_extra = tr.Compose([self.transforms, augmentations])

            if dataset == "MNIST":
                self.trainset_extra = MNIST(root="data", train=True, download=True, transform=self.transforms_extra)
            elif dataset == "FashionMNIST":
                self.trainset_extra = FashionMNIST(root="data", train=True, download=True, transform=self.transforms_extra)

            x_train_extra = self.trainset.data.reshape(-1, 28*28)/255.
            y_train_extra = self.trainset.targets
            train_data_extra = TensorDataset(x_train_extra, y_train_extra)

            self.train_data = ConcatDataset([self.train_data, train_data_extra])


    def get_data(self, num_train_samples=None):
        if num_train_samples is not None:
            sub_train_idx = np.random.choice(self.train_data.__len__(), num_train_samples, replace=False)
            sub_train_data = Subset(self.train_data, sub_train_idx)

        else:
            sub_train_data = self.train_data

        return sub_train_data, self.test_data


    



    



    