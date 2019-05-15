import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms


class SimpleNet(nn.Module):
    def __init__(self, n_classes=10):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.GAP(F.relu(self.conv4(x)))
        x = x.view(-1, 128)
        x = self.fc(x)
        return x

class Tester():
    """ based off https://karpathy.github.io/2019/04/25/recipe/
    
    This is an attempt to make a model train resonably well, to check for bugs in intialization, gradients, etc. 
    NOT a way to find if a model is appropriate for a given dataset. Of course these questions are not independent,
    but even a poor model for a dataset should pass these "tests"
    
    Not for data exploration. If you need this check:
    https://pair-code.github.io/what-if-tool/
    
    
    ADD THIS: https://pytorch.org/docs/stable/tensorboard.html. Use this for all important tests
    
    Possible: compute the mean / std of a few examples from a batch, print them out to see if fits with dataloader
    
    Add a trining thing to try a bunch of different learning rates to see which trains best - then sent to tensorboard???
    Add model.train(), model.eval()
    
    for baseline use adam or SGD? bn or not? probably dont need it if net is small
    
    (maybe two settings-one normal, and one extensive where different optimizers are tried. But maybe this isn't for optimization,
    we just want something to work. So adam with .1, .01, .001, .0001)
    
    
    Maybe make a self.optim, self.loss, so it isn t suplicated every time in multiple functions
    maybe should be raising Error or except AssertionError: for these things
    
    where should the nan / inf check go? it could be useful throughout training steps.
    A lot of these tests are just ways to simply the full through debugging that you'd do using tensorboard
    
    Possibly switch to using pandas df for the printing parameters and inf/nan. 
    """
    def __init__(self, model,seed=101, really_deterministic=False, toy_dataset='cifar10', loss='cross_entropy'):
#         self.dataset = 'cifar'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if really_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        self.model = model.to(self.device)
        if loss == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()        
        lr =.002 # ??? self.find_lr??? 
        # maybe a different optimizer for sanity checks vs. overfitting? 
        # like do automate lr finding for actually training the model, .002 for making sure model works.
        # is there a problem for not reinitializing the optimizers each time?
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.toy_dataset = toy_dataset
        if self.toy_dataset == 'cifar10':
            self.toy_train_loader, self.toy_val_loader = self.get_cifar_loaders(bs=4)
            


    def get_cifar_loaders(self, bs=128, size=32, PATH='/media/rene/data/', augmentation=False):
        aug_transform = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                 (0.2023, 0.1994, 0.2010)),
                                            ]) 
        no_aug_transform = transforms.Compose(
                                              [transforms.Resize(size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                    (0.2023, 0.1994, 0.2010))
                                               ])
        if augmentation:
            train_transform = aug_transform
        else:
            train_transform = no_aug_transform

        trainset = torchvision.datasets.CIFAR10(root=PATH, train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                  shuffle=True, num_workers=4)

        valset = torchvision.datasets.CIFAR10(root=PATH, train=False,
                                               download=True, transform=no_aug_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=bs,
                                                 shuffle=False, num_workers=4)
        return train_loader, val_loader

#     def get_dataset_baselines(self, dataset):
#         """Get performance of baseline models for comparision. 
#         Plot training progress of the three models to know they actually converged."""


##### General stuff



    
#### Make sure gradients exist in right places and parameters are updated
    def print_non_trainable_params(self, print_all=False):
        """First sanity check: If params aren't trainable, there's no hope."""
        self.model = self.model.train()
        
        if print_all:
            for name, param in self.model.named_parameters():
                print(name, param.size())
        else:
            i = 0
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    print(name, param.size())
                    i+=1
            if i ==0:
                print("All parameters require grad")
        
    def check_correct_params_updated(self):
        """During training, trainable params should change and non-trainable shouldn't"""
        trainable_params = [param for param in self.model.named_parameters() if param[1].requires_grad]
        initial_trainable_params = [(name, p.clone()) for (name, p) in trainable_params]
        non_trainable_params = [param for param in self.model.named_parameters() if not param[1].requires_grad]
        initial_non_trainable_params = [(name, p.clone()) for (name, p) in non_trainable_params]
        
        bs = 32
        steps=10
        self.model.train()
        
        x, y = next(iter(self.toy_train_loader))
        x, y = x.to(self.device), y.type(torch.LongTensor).to(self.device)

        for i in range(steps):
            output = self.model(x)
            loss = self.loss_fn(output, y)
            _, predicted = output.max(1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        not_updated_trainable_params = []
        for (_, inital_param), (name, curr_param) in zip(initial_trainable_params, trainable_params):
            if torch.equal(inital_param, curr_param):
                  not_updated_trainable_params.append(name)
                  
        updated_non_trainable_params = []
        for (_, inital_param), (name, curr_param) in zip(initial_non_trainable_params, non_trainable_params):
            if(not torch.equal(inital_param, curr_param)):
                  updated_non_trainable_params.append(name)
        
        if ((len(not_updated_trainable_params)>0) or (len(updated_non_trainable_params)>0)):
            print(f'Something is wrong. These traininable params were not updated:', *not_updated_trainable_params, sep="\n")
            print(f'These non-trainable params were updated:', *updated_non_trainable_params, sep="\n")
        else:
            print("All correct params updated or not updated")
                        
            
#### Test model being resonable / gradient is actually correct

    def check_batch_gradients_mixed(self):
        """Too easy to accidentally mix batch dimension with data dimensions.
        Make loss depend on only one example in output
        Ensure that the gradient is only non zero for this example in input
        """
        
        x, y = next(iter(self.toy_train_loader))
        x, y = x.to(self.device), y.type(torch.LongTensor).to(self.device)
        x.requires_grad = True

        pred = self.model(x)
        loss = self.loss(pred[0:1, :], y[0:1])
        loss.backward()
        
        print(x.grad[0:1, :, :, :])
        print(x.grad[1:, :, :, :])
        
        if ((len(torch.nonzero(x.grad[0:1, :, :, :])) > 1) and 
            (len(torch.nonzero(x.grad[1:, :, :, :])) == 0)):
            print('Gradient for the correct inputs only') 
        else:
            grad_dims_size = np.prod(torch.tensor(x.grad[0:1, :, :, :].size()).detach().cpu().numpy())
            grad_dims_nonzero_frac = len(torch.nonzero(x.grad[0:1, :, :, :]))/grad_dims_size
            
            other_dims_size = np.prod(torch.tensor(x.grad[1:, :, :, :].size()).detach().cpu().numpy())
            other_dims_zero_frac = len(torch.nonzero(x.grad[1:, :, :, :]))/other_dims_size
            
            print(f'Something is wrong. Non-zero grads are not only in correct input')
            print(f'Frac non-zero grads in correct inputs: {grad_dims_nonzero_frac}')
            print(f'Frac zero grads in other inputs: {other_dims_zero_frac}')
            
            
    def print_tensor_stats(self, x):
        perc_unique = 100*len(torch.unique(x)) / torch.prod(torch.tensor(x.size()))
        
        print(f'{type(x)}, Mean: {torch.mean(x):6.4f}, std: {torch.std(x):6.4f}, % Unique: '
              f'{perc_unique:6.4f}, Size: {torch.tensor(x.size()).detach().cpu().numpy()}')
        
    def intermediate_output_stats(self):
        """For when the output looks weird, see where net is going wrong"""
        
        for name, param in self.model.named_parameters():
            param.register_hook(lambda x: self.print_tensor_stats(x))
                
        x, y = next(iter(self.toy_train_loader))
        x, y = x.to(self.device), y.type(torch.LongTensor).to(self.device)
        output = self.model(x)
            
            
    def check_loss_sensible(self):
        self.model.train()
        x, y = next(iter(self.toy_train_loader))
        x, y = x.to(self.device), y.type(torch.LongTensor).to(self.device)
        output = self.model(x)
        loss = self.loss_fn(output, y)
        
        if not torch.isfinite(output).all():
            raise Exception(f'Output is NaN/Inf. Outputs: {output.detach().cpu().numpy()}')
        else:
            print('All outputs are finite')
        if type(self.loss_fn) is type(nn.CrossEntropyLoss()):
            expected_loss = -math.log(1/(output.size()[1]))

        if not .1 < abs(expected_loss)/abs(loss) < 10:
            print(f'Loss is strange. Expected loss with random init: {expected_loss :6.4f}, Curr loss: {loss :6.4f}')
        else:
            print(f'Expected loss with random init: {expected_loss:6.4f}, Curr loss: {loss:6.4f}')
            print(f'Initial loss is resonable')
            print(f'Outputs: {output.detach().cpu().numpy()}')
            self.intermediate_output_stats()
            

        
            
    def check_weights_nan_inf(self):
        for name, param in self.model.named_parameters():
            perc_finite = 100.0*torch.sum(torch.isfinite(param).detach())/ \
                            np.prod(torch.tensor(param.size()).detach().cpu().numpy())

            if perc_finite<100:
                perc_nan = 100.0*torch.sum(torch.isnan(param).detach())/ \
                                np.prod(torch.tensor(param.size()).detach().cpu().numpy())
                perc_inf = 100.0*torch.sum(torch.isinf(param).detach())/ \
                                np.prod(torch.tensor(param.size()).detach().cpu().numpy())
                print(f'Param: {name: <25}: {perc_finite: <3}% Finite, with: '
                      f'{perc_nan: <3}% NaN, {perc_inf: <3}% Inf, with size: '
                      f'{torch.tensor(param.size()).detach().cpu().numpy()}')
            else:
                print(f'Param: {name: <25}:  All are finite')
            
            
    def check_grad_nan_inf(self):
        """Checking grads after a single backward pass"""
        self.model.train()
        
        x, y = next(iter(self.toy_train_loader))
        x, y = x.to(self.device), y.type(torch.LongTensor).to(self.device)

        output = self.model(x)
        loss = self.loss_fn(output, y)
        self.optimizer.zero_grad()
        loss.backward()

        print('loss', loss.item())
        
        for name, param in self.model.named_parameters():
            grad = param.grad
            perc_finite = 100.0*torch.sum(torch.isfinite(grad).detach())/ \
                            np.prod(torch.tensor(grad.size()).detach().cpu().numpy())
            print(perc_finite)
            print(grad)

            if perc_finite<100:
                perc_nan = 100.0*torch.sum(torch.isnan(grad).detach())/ \
                                np.prod(torch.tensor(grad.size()).detach().cpu().numpy())
                perc_inf = 100.0*torch.sum(torch.isinf(grad).detach())/ \
                                np.prod(torch.tensor(grad.size()).detach().cpu().numpy())

                print(f'Grad: {name: <25}: {perc_finite: <3}% Finite, with: '
                      f'{perc_nan: <3}% NaN, {perc_inf: <3}% Inf, with size: '
                      f'{torch.tensor(param.size()).detach().cpu().numpy()}')
            else:
                print(f'Grad: {name: <25}:  All are finite')
            
            


        
    def big_check(self):
        """The goal is to check these somewhat independent things:

        1. Are params are trainable?
        2. Are all the trainable params actually updated during training?
        
        2. Does this training do something sensible at all (overfit a small dataset)
        3. Have we made the annoying common mistake of mixing data and channel dimensions?
        
        
        Maybe two setttings: One simple overfit?? 
        
        Or the other would be to throughly check everything. The issue is overfit isn't enough, could
        still be some layers not training. 
        
        At what 0% inf is there an issue?
        
        """
        
        #### Checks for really dumb mistakes:
        # these are all important sanity checks. Possibly could conditionally exclude 
        self.print_non_trainable_params()
        self.check_correct_params_updated()
        
        # possibly exclude this if others work:
        self.check_batch_gradients_mixed()
        
        ### Now to check if more annoying issues are present:
        
        # maybe should be first, but because graph 
        self.overfit_one_batch()
            
        # this needs to be in a longer training loop, and maybe add recording the number of nans each time:
        self.check_nan_inf()


#     def train_epoch(self, train_loader, model, criterion, optimizer):    
#         """Train for 1 epoch. Return acc and avg. loss per img."""
#         model.to(self.device)
#         model.train()

#         loss = 0
#         correct = 0
#         total = 0

#         for i, (input, target) in enumerate(train_loader):
#             input, target = input.to(self.device), target.type(torch.LongTensor).to(self.device)
#             output = model(input)
#             loss = criterion(output, target)

#             loss += loss.item()
#             _, predicted = output.max(1)
#             total += target.size(0)
#             correct += predicted.eq(target).sum().item()

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         return correct/total, loss/total


    def overfit_one_batch(self, dataloader='cifar10'):
        """Check if network can overfit a single batch of 2 examples.
        Loss should go to 0, otherwise model is not complex
        
        ***displays a graph of the loss as well as the 
        """
        bs = 2
        steps=200
        lr =.003 # ??? self.find_lr???
        
        if dataloader == 'cifar10':
            print('Using CIFAR10 dataloader with bs=10 for overfitting')
            train_loader, val_loader = self.get_cifar_loaders(bs=bs)
            
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(steps/3), gamma=0.2)
        
        train_loss = []
        train_acc = []
        x, y = next(iter(train_loader))
        x, y = x.to(self.device), y.type(torch.LongTensor).to(self.device)

        for i in range(steps):
            output = self.model(x)
            loss = loss_fn(output, y)

            train_loss.append(loss.item())
            _, predicted = output.max(1)
            train_acc.append(predicted.eq(y).sum().item()/bs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
                        
        plt.style.use('seaborn-colorblind')
        fig, ax1 = plt.subplots(figsize=(8,5))   
        ax1.set_xlabel('Step', fontsize='x-large')
        ax1.set_ylabel('Loss', fontsize='x-large', color='tab:red')
        ax1.plot(list(range(steps)), train_loss, label="Overfitting Train Loss",
                 linewidth=2, color='tab:red')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Acc', fontsize='x-large', color='tab:blue')
        ax2.plot(list(range(steps)), train_acc, label="Overfitting Train Acc", 
                 linewidth=2, color='tab:blue')
        ax2.legend(loc='center right', prop={'size': 10})
        plt.title('Overfitting One Batch')
        fig.tight_layout() 
        plt.show()

        
#### Testing trainied net features
        
    def nonsense_inputs(self):
        """Performance on inputs of 0, 1, and Normal(0, 1)"""
    
    
    def custom_dataset():
        """visualize batch (before net & w / wo aug), 
        functions to look at the whole dataset and see class imbalances, etc."""
    
#### Testing trainied net features
    def vis_prediction_dynamics():
        """for a fixed batch, vis predictions during training."""