"""
This script runs the experiment for our project.

"""

# Importing libraries ---------------------------------------------------------

import numpy as np
import torch
import torchvision.transforms as tr
import pandas as pd

from data import Data
from priors import *
from Networks import *
from BayesianNN import BNN_MCMC

# Setting seeds ---------------------------------------------------------------
torch.manual_seed(1)


# Specify the prior -----------------------------------------------------------

# Possible prior choices: 
#       Isotropic_Gaussian, 
#       StudentT_prior
#       Laplace_prior
#       Gaussian_Mixture
#       Normal_Inverse_Gamma
#       GaussianSpikeNSlab
#       MixedLaplaceUniform

prior = Normal_Inverse_Gamma()


# Specify the iteration parameters --------------------------------------------

# network list
networks = {"FCNN": FullyConnectedNN(), 
        #"CNN": ConvolutionalNN()
        }

# Temperature list
Temperatures = [0, 0.001, 0.01, 0.1, 1.]


# sample size list
sample_sizes = [3750, 15000, 60000, 120000]

# preallocate pandas dataframe for results
results = pd.DataFrame(columns = [
    "Network", 
    "Sample Size", 
    "Epochs", 
    "Burn in", 
    "sample interval", 
    "Temperature", 
    "Test Accuracy", 
    "Test ECE", 
    "Test AUROC",
    "Posterior mean",
    "Posterior var"],
    index = range(len(networks)*len(Temperatures)*len(sample_sizes)))


#create a dict for the different parameter values
base_epoch, base_burn_in, base_sample_interval, base_samplesize = 50, 10, 2, sample_sizes[-1]
args_dict = [(sample_size, (base_epoch*base_samplesize/sample_size, base_burn_in*base_samplesize/sample_size, base_sample_interval*base_samplesize/sample_size )) for sample_size in sample_sizes]
args_dict = dict(args_dict)

# Run the experiment ----------------------------------------------------------

iteration = 0

for net in networks.keys():
    for T in Temperatures:
        for n in range(len(sample_sizes)):
        
            # print iteration info
            print(50*"-")
            print("Iteration: ", iteration + 1, " of ", len(networks)*len(Temperatures)*len(sample_sizes))
            print("Network:     ", net)
            print("Prior:       ", prior.name)
            print("Temperature: ", T)
            """
            print("Sample size: ", sample_sizes[n])
            print("Epoch:       ", args_dict[sample_sizes[n]][0])
            print("Burn in:     ", args_dict[sample_sizes[n]][1])
            print("Sample interval: ", args_dict[sample_sizes[n]][2])
            """

            # get data
            if sample_sizes[n] == 120000:
                # if sample size is 120000, use data augmentation
                augmentations = tr.Compose([tr.RandomRotation(5)])
                train_data, test_data = Data("MNIST", augmentations = augmentations).get_data()
            else:
                # subsample original train data if sample size is smaller than 120000
                train_data, test_data = Data("MNIST", augmentations = None).get_data(num_train_samples=sample_sizes[n])


            # run BNN
            model = BNN_MCMC(
                train_data,
                network = networks[net],
                prior=prior,
                Temperature = T,
                num_epochs = int(args_dict[sample_sizes[n]][0]),
                max_size = 20,
                burn_in = int(args_dict[sample_sizes[n]][1]),
                lr = 1e-3,
                sample_interval = int(args_dict[sample_sizes[n]][2]))

            model.train()

            # get test metrics
            acc, ece, auroc = model.get_metrics(test_data)
            post_mean, post_var = model.get_posterior_stats()

            #print("Test accuracy: ", acc)
            #print("Test ECE: ", ece)
            #print("Test AUROC: ", auroc)

            # save results
            results.iloc[iteration, :] = net, sample_sizes[n], args_dict[sample_sizes[n]][0], args_dict[sample_sizes[n]][1], args_dict[sample_sizes[n]][2], T, acc, ece, auroc, post_mean, post_var
            iteration += 1

# save results to csv
results.to_csv(f"results/results_{prior.name}.csv")