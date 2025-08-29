# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 23:18:32 2025

@author: teowe
"""

from torchvision import datasets, transforms
import torch
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Sequence
import torch.distributed as dist
import time
from torch.utils.data import random_split
#working_directory = "C:\Masters\MSc Project\MNIST"
#os.chdir(working_directory)

from decor import DecorLinear, DecorConv2d
import adamw_eg
import init_tools
import numpy as np
import wandb
import argparse
import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="Training Loop Example")
parser.add_argument('--dataset')
parser.add_argument('--lr', type=float)
parser.add_argument('--dlr', type=float)
parser.add_argument('--decay', type=float)
parser.add_argument('--seed', type=int)
parser.add_argument('--folder', type=str)
parser.add_argument('--EG', type=str, default="false")
parser.add_argument('--wandb', type=str)
parser.add_argument('--bias', type=str, default="false")
args = parser.parse_args()

seed = args.seed # pick any fixed number
#args.seed
folder = args.folder

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
start_time = time.time()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(seed)
wandb.login(key="8a19d6b67d9330154ac4847a0c52126f6cfba374")

class BPLinear(torch.nn.Linear):
    """BP Linear layer"""

    def __str__(self):
        return "BPLinear"
    
class BPConv2d(torch.nn.Conv2d):
    """BP Conv2d layer"""

    def __str__(self):
        return "BPConv2d"

from torchvision import datasets, transforms
import torch


    
# Specify a different root directory
root_dir = './data'
full_train = datasets.MNIST(root=root_dir, train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root=root_dir, train=False, transform=transforms.ToTensor(), download=True)
train_data, val_data = random_split(full_train, [50000, 10000])
print(len(train_data))
print(len(val_data))


import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# During DataLoader creation:
#train_sampler = DistributedSampler(train_data)
#test_sampler = DistributedSampler(test_data)
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=32, 
    shuffle=True,
    #sampler = test_sampler,
    #generator=torch.Generator(device='cuda'),
    num_workers=0,
    generator=g
)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=32, 
    shuffle=True,
    #sampler = train_sampler,
    #generator=torch.Generator(device='cuda'),
    num_workers=0,
    generator=g
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32, 
    shuffle=True,
    #sampler = train_sampler,
    #generator=torch.Generator(device='cuda'),
    num_workers=0,
    generator=g
)




data = np.load('./K/kmnist-train-imgs.npz')


class NPZDataset(Dataset):
    def __init__(self, npz_file_images, npz_file_labels):
        images = np.load(npz_file_images)
        labels = np.load(npz_file_labels)
        self.images = torch.tensor(images['arr_0'], dtype=torch.float32)
        self.labels = torch.tensor(labels['arr_0'], dtype=torch.long)
        
        # Normalize and reshape if needed
        if self.images.ndim == 3:
            self.images = self.images.unsqueeze(1)  # (N, 1, 28, 28)
        self.images /= 255.0  # normalize to [0, 1]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Usage
k_full_train = NPZDataset('./K/kmnist-train-imgs.npz', './K/kmnist-train-labels.npz')
k_test_dataset = NPZDataset('./K/kmnist-test-imgs.npz', './K/kmnist-test-labels.npz')
k_train_dataset, k_val_dataset = random_split(k_full_train, [50000, 10000])


#k_train_sampler = DistributedSampler(k_train_dataset)
#k_test_sampler = DistributedSampler(k_test_dataset)

k_train_loader = torch.utils.data.DataLoader(
    k_train_dataset,
    batch_size=32,
    shuffle=True,
    #sampler = k_train_sampler,
    #generator=torch.Generator(device='cuda'),
    num_workers=0,
    generator=g
)

k_val_loader = torch.utils.data.DataLoader(
    k_val_dataset,
    batch_size=32,
    shuffle=True,
    #sampler = k_test_sampler,
    #generator=torch.Generator(device='cuda'),
    num_workers=0,
    generator=g
)

k_test_loader = torch.utils.data.DataLoader(
    k_test_dataset,
    batch_size=32,
    shuffle=True,
    #sampler = k_test_sampler,
    #generator=torch.Generator(device='cuda'),
    num_workers=0,
    generator=g
)


import torch
import torch.nn as nn
from decor import DecorLinear, DecorConv2d

class DecorMLP(nn.Module):
    def __init__(self, decor):
        super().__init__()
        self.DecorLinear1 = DecorLinear(BPLinear, 784, 500, decor_lr = decor)
        self.DecorLinear2 = DecorLinear(BPLinear, 500, 500, decor_lr = decor)
        self.DecorLinear3 = DecorLinear(BPLinear, 500, 500, decor_lr = decor)
        self.DecorLinear4 = DecorLinear(BPLinear, 500, 10, decor_lr = decor)
        self.relu = nn.ReLU()
    
    def forward(self, x, return_hidden=False):
        x = x.view(x.size(0), -1)            
        x1 = self.relu(self.DecorLinear1(x))    
        x2 = self.relu(self.DecorLinear2(x1))
        x3 = self.relu(self.DecorLinear3(x2))
        x4 = self.DecorLinear4(x3)
        if return_hidden:
            return x3  # return activations before final layer
    
        
        return x4


# class DecorMLP(nn.Module):
#     def __init__(self, decor):
#         super(DecorMLP, self).__init__()
#         self.net = nn.Sequential(            
#         DecorLinear(BPLinear, 784, 500, decor_lr = decor),
#         nn.ReLU(),
#         DecorLinear(BPLinear, 500, 500, decor_lr = decor),
#         nn.ReLU(),
#         DecorLinear(BPLinear, 500, 500, decor_lr = decor),
#         nn.ReLU(),
#         DecorLinear(BPLinear, 500, 10, decor_lr = decor)     
#         )

    
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         return self.net(x)



import torch
import torch.nn as nn
from decor import DecorLinear, DecorConv2d

class MLP(nn.Module):
    def __init__(self, layer_kwargs={} ):
        super(MLP, self).__init__()
        self.Linear1 = nn.Linear(784,500)
        self.Linear2 = nn.Linear(500,500)
        self.Linear3 = nn.Linear(500,500)
        self.Linear4 = nn.Linear(500,10)
        self.relu = nn.ReLU()
    
    def forward(self,x, return_hidden=False):
        x = x.view(x.size(0), -1)  
        x1 = self.relu(self.Linear1(x))
        x2 = self.relu(self.Linear2(x1))
        x3 = self.relu(self.Linear3(x2))
        x4 = self.Linear4(x3)
        if return_hidden:
            return x3  # return activations before final layer
    
        
        return x4


# class MLP(nn.Module):
#     def __init__(self, layer_kwargs={}):
#         super(MLP, self).__init__()
#         self.net = nn.Sequential(            
#         nn.Linear(784,500),
#         nn.ReLU(),
#         nn.Linear(500,500),
#         nn.ReLU(),
#         nn.Linear(500,500),
#         nn.ReLU(),
#         nn.Linear(500,10) 
#         )

    
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         return self.net(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()
print(device)
#TO COMPLETE --> Running your CNN model class
from torch import Tensor
import numpy as np
from torch import nn, optim

def stats(loader, net):
        correct = 0
        total = 0
        running_loss = 0
        n = 0    # counter for number of minibatches
        with torch.no_grad():
            for data in loader:
                #print(data)
                images, labels = data
            # images, labels = images.to(device), labels.to(device)
                
                
                outputs = net(images)   
        
                # accumulate loss
                running_loss += loss_fn(outputs.cpu(), labels.cpu())
                n += 1
                
                # accumulate data for accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)    # add in the number of labels in this minibatch
                correct += (predicted.cpu() == labels.cpu()).sum().item()  # add in the number of correct labels
                
        return running_loss/n, correct/total 

import torch



def train(name,testloader,trainloader,learning_rate, decor_learning_rate, decay):
    final_epoch_hidden = []
    final_epoch_labels = []
    project = args.wandb
    wandb.init(
    project= f"{project}",
    name=name, 
    config={
        "learning_rate": f"{learning_rate:.0e}",
        "decor learning rate": f"{decor_learning_rate:.0e}",
        "weight decay": f"{decay:.0e}",
        "epochs": 50,
        "batch size": 32
    }
)
    print("trainingg-----------------------------------------------------------------------")
    
    print(name)
    if(decor_learning_rate == 0): 
        print("using normal MLP")
        #torch.distributed.init_process_group(backend='nccl')
        model = MLP()
        model.to(device) 
        net = model
        net = net 
        net.to(device) 
        
    else:
        print("using DecorMLP")
        #torch.distributed.init_process_group(backend='nccl')
        model = DecorMLP(decor_learning_rate)
        model.to(device) 
        net = model
        net = net 
        net.to(device) 
        if(args.bias == "true"):
            init_tools.set_split_bias(net)
    
    
    
    nepochs = 50
    results_path = f"/scratch/tcrv4423/{folder}/results/{seed}/{name}.pt"

    statsrec = np.zeros((6,nepochs))
    #lval = np.zeros(nepochs) 
    #aval = np.zeros(nepochs) 

    
    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    if(args.EG == "false"):
        print("no eg")
        optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay = decay)
    elif(args.EG == "true"):
        print("eg")
        optimizer = adamw_eg.AdamWeg(net.parameters(), lr=learning_rate, weight_decay = decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=nepochs)
    for epoch in range(nepochs):  # loop over the dataset multiple times
        # if isinstance(trainloader.sampler, torch.utils.data.distributed.DistributedSampler):
        #     trainloader.sampler.set_epoch(epoch)
        correct = 0          # number of examples predicted correctly (for accuracy)
        total = 0            # number of examples
        running_loss = 0.0   # accumulated loss (for mean loss)
        n = 0                # number of minibatches
        top1_train_acc = 0.0
        top5_train_acc = 0.0
        val_loss = 0.0 
        for data in trainloader:
            net.to(device) 
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            #outputs.to("")
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward, backward, and update parameters
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
            # accumulate loss
            running_loss += loss.item()
            n += 1

            # Compute Top-1 and Top-5 Accuracy
            # top1, top5 = topk_accuracy(outputs, labels, topk=(1, 5))
            # top1_train_acc += top1.item()/ len(train_loader)
            # top5_train_acc += top5.item()/ len(train_loader)
            
            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels
        
        ltrn = running_loss/n
        atrn = correct/total 
        #net.to("cpu")
        #lval, aval = stats(val_loader, net)

        net.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0.0
        total = 0.0
        n_val = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)  # Forward pass
                loss = loss_fn(outputs, labels)  # Compute loss
                val_loss += loss.item()
                n_val += 1


                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)    # add in the number of labels in this minibatch
                correct += (predicted == labels).sum().item()  # add in the number of correct labels
                #print(len(inputs))
                
            aval = round(correct/total, 3)
            lval = val_loss / n_val
        scheduler.step(lval)
        net.train()
                

        # collect together statistics for this epoch
        statsrec[:,epoch] = (ltrn, atrn, top1_train_acc, top5_train_acc, lval, aval)
        wandb.log({
            "train_loss": ltrn,
            "train_accuracy": atrn,
            "train_top5_accuracy": top5_train_acc,
            "val_loss": lval,
            "val_accuracy": aval
        }, step=epoch)
        print(
            f"Epoch {epoch+1}/{nepochs} | "
            f"Train Loss: {ltrn:.3f} | Train Acc: {atrn:.1%} | "
            f"Validation Loss: {lval:.3f} | " 
            f"Validation Acc: {aval:.1%}% | " 

        )
        
    torch.save({"state_dict": net.state_dict(), "stats": statsrec}, results_path)

        


    


dataset = args.dataset
lr = args.lr
dlr = args.dlr
decay = args.decay
if(dlr == 0):
    if(dataset == "reg"):
        name = f"cor-reg{lr:.0e},{dlr:.0e},{decay:.0e},{seed}"
        training_loader = train_loader
        testing_loader = val_loader
    elif(dataset == "k"):
        name = f"cor-k{lr:.0e},{dlr:.0e},{decay:.0e},{seed}"
        training_loader = k_train_loader
        testing_loader = k_val_loader
    
else:
    if(dataset == "reg"):
        name = f"decor-reg{lr:.0e},{dlr:.0e},{decay:.0e},{seed}"
        training_loader = train_loader
        testing_loader = val_loader
    elif(dataset == "k"):
        name = f"decor-k{lr:.0e},{dlr:.0e},{decay:.0e},{seed}"
        training_loader = k_train_loader
        testing_loader = k_val_loader


# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#train("decor", k_test_loader, k_train_loader, 1e-4, 1e-5, 1e-7)
train(name, testing_loader, training_loader, lr, dlr, decay)
    
    
    
    



total_time = time.time() - start_time
wandb.finish()
print(f"Total training runtime: {total_time:.2f} seconds")

import torch

# Assume you already collected X (activations) and Y (targets)
# X: shape (n_samples, d)
# Y: shape (n_samples, c) or (n_samples,)




    
    
    
    
    
    
    
