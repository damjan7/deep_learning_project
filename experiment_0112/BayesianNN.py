""" Bayesian Neural Network """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import PolynomialLR
from torchmetrics.functional import calibration_error
from collections import deque, OrderedDict
from tqdm import trange
import copy
from sklearn.metrics import roc_auc_score
from SGLD import SGLD


class BNN_MCMC:
    def __init__(self, dataset_train, network, prior, Temperature = 1.,
     num_epochs = 300, max_size = 100, burn_in = 100, lr = 1e-3, sample_interval = 1, device = "cpu"):
        super(BNN_MCMC, self).__init__()

        # set device 
        self.device = torch.device(device)

        # Hyperparameters and general parameters
        self.Temperature = Temperature
        self.learning_rate = lr
        self.num_epochs = num_epochs
        self.burn_in = burn_in
        self.sample_interval = sample_interval
        self.max_size = max_size


        self.batch_size = 128
        self.print_interval = 50
        
        # Data Loader
        self.data_loader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        self.sample_size = dataset_train.__len__()

        # Set Prior
        self.prior = prior

        # Initialize the network
        self.network = network.to(self.device)

        # Set optimizer
        self.optimizer = SGLD(self.network.parameters(), lr=self.learning_rate, num_data=self.batch_size, temperature=self.Temperature)

        # Scheduler for polynomially decreasing learning rates
        self.scheduler = PolynomialLR(self.optimizer, total_iters = self.num_epochs, power = 0.5)

        # Deque to store model samples
        self.model_sequence = deque()

    def train(self):
        num_iter = 0
        print('Training Model')

        self.network.train()
        progress_bar = trange(self.num_epochs)

        N = torch.tensor(self.sample_size, device = self.device)
        if self.prior.name == 'Normal Inverse Gamma':
            n_params = 0
            SS_params = 0

        for _ in progress_bar:
            num_iter += 1

            for batch_idx, (batch_x, batch_y) in enumerate(self.data_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                self.network.zero_grad()
                n = len(batch_x)

                # Perform forward pass
                current_logits = self.network(batch_x)

                # Compute the NLL
                nll = N/n*F.nll_loss(F.log_softmax(current_logits, dim=1), batch_y)

                # Compute the log prior
                log_prior = torch.tensor(0,device=self.device, dtype=torch.float32)

                # prior for Normal Inverse Gamma
                if self.prior.name == 'Normal_Inverse_Gamma':
                    param_list = torch.tensor([],device=self.device)
                    for name, param in self.network.named_parameters():
                        if param.requires_grad:
                            param_list = torch.cat((param_list, param.view(-1)))
                            
                    current_var = torch.var(param_list)
                    log_prior += self.prior.log_likelihood(param, current_var).sum()

                else:
                    for name, param in self.network.named_parameters():
                        if param.requires_grad:
                            # param=torch.tensor(param,device=self.device)
                            log_prior += self.prior.log_likelihood(param).sum()
                

                # Calculate the loss
                #loss = N/n*F.nll_loss(F.log_softmax(current_logits, dim=1), batch_y) - log_prior
                loss = nll - log_prior

                # Backpropagate to get the gradients
                loss.backward(retain_graph=True)

                # Update the weights
                self.optimizer.step()

                # Update Metrics according to print_interval
                if batch_idx % self.print_interval == 0:
                    current_logits = self.network(batch_x)
                    current_accuracy = (current_logits.argmax(axis=1) == batch_y).float().mean()
                    progress_bar.set_postfix(loss=loss.item(), acc=current_accuracy.item(),
                    nll_loss=N/n*F.nll_loss(F.log_softmax(current_logits, dim=1), batch_y).item(),
                    log_prior_normalized = - log_prior.item(),
                    lr = self.optimizer.param_groups[0]['lr'])

            # Decrease lr based on scheduler
            self.scheduler.step()
            
            # Save the model samples if past the burn-in epochs according to sampling interval
            if num_iter > self.burn_in and num_iter % self.sample_interval == 0:
                self.model_sequence.append(copy.deepcopy(self.network))
                # self.network.state_dict()

            # If model_sequence to big, delete oldest model
            if len(self.model_sequence) > self.max_size:
                self.model_sequence.popleft()

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        self.network.eval()

        # Sum predictions from all models in model_sequence
        estimated_probability = torch.zeros((len(x), 10), device = self.device)

        for model in self.model_sequence:

            self.network.load_state_dict(model.state_dict())
            logits = self.network(x).detach()
            estimated_probability += F.softmax(logits, dim=1)
        
        # Normalize the combined predictions to get average predictions
        estimated_probability /= len(self.model_sequence)

        assert estimated_probability.shape == (x.shape[0], 10)  
        return estimated_probability
    
    def test_accuracy(self,x):
        # test set
        x_test = x[:][0].clone().detach().to(self.device)
        y_test = x[:][1].clone().detach().to(self.device)      

        # predicted probabilities
        class_probs = self.predict_probabilities(x_test)

        # accuracy
        accuracy = (class_probs.argmax(axis=1) == y_test).float().mean()
        return  accuracy.cpu().numpy()

    def test_calibration(self,x):
        # test set
        x_test = x[:][0].clone().detach().to(self.device)
        y_test = x[:][1].clone().detach().to(self.device)       

        # predicted probabilities
        class_probs = self.predict_probabilities(x_test)

        calib_err = calibration_error(class_probs, y_test, n_bins = 30, task = "multiclass", norm="l1", num_classes=10)
        return calib_err.cpu().numpy()

    # def test_auroc(self,x):
    #     # test set
    #     x_test = x[:][0].clone().detach()
    #     y_test = x[:][1].clone().detach()         

    #     # predicted probabilities
    #     class_probs = self.predict_probabilities(x_test)

    #     auroc = roc_auc_score(y_test, class_probs, multi_class='ovr')
    #     return auroc

    def get_metrics(self, x):
        accuracy = self.test_accuracy(x)
        calib_err = self.test_calibration(x)
        auroc = self.test_auroc(x)

        return accuracy, calib_err, auroc

    def get_posterior_stats(self):
        self.network.eval()

        # get weights from all models
        param_flat_all = torch.tensor([],device = self.device)
        for model in self.model_sequence:
            parameters = model.state_dict()
            param_values = list(parameters.values())
            param_flat = torch.cat([v.flatten() for v in param_values])
            param_flat_all.append(param_flat.flatten())

        param_flat_all = torch.cat(param_flat_all)

        # get mean and variance
        mean = torch.mean(param_flat_all, dim=0)
        var = torch.var(param_flat_all, dim=0)


        return mean.cpu().numpy(), var.cpu().numpy()