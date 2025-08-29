from pathlib import Path
from torchvision import datasets, transforms
import torch
import torch.optim
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Sequence
import torch.distributed as dist
import time
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import wandb
import seaborn as sns
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import math
from typing import List
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
from ffcv.transforms import NormalizeImage
import torchvision.transforms
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
#working_directory = "C:\Masters\MSc Project\CIFAR"
#os.chdir(working_directory)

from decor import DecorLinear, DecorConv2d
import adamw_eg
import init_tools



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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = args.seed# pick any fixed number
#args.seed
folder = args.folder

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(seed)
wandb.login(key="8a19d6b67d9330154ac4847a0c52126f6cfba374")
start_time = time.time()

# Note that statistics are wrt to uin8 range, [0,255].
dataset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=False)
data = np.stack([np.array(img) for img, _ in dataset])  # shape: (50000, 32, 32, 3)

# Compute mean and std per channel
mean = np.array(data.mean(axis=(0, 1, 2)))   # shape: (3,)
std = np.array(data.std(axis=(0, 1, 2)) )     # shape: (3,)

print("Mean:", mean)
print("Std:", std)
print(len(mean))
# Mean: [125.30691805 122.95039414 113.86538318]
# Std: [62.99321928 62.08870764 66.70489964]
train_mean = [125.23070598, 122.87243025, 113.82591189]
train_std = [62.99841571, 62.05788563, 66.68000432]

val_mean = [125.61176631, 123.26224971, 114.02326836]
val_std = [62.9715069,  62.21086592, 66.80415493]

test_mean = [126.02464141, 123.7085042,  114.85431865]
test_std = [62.89639135, 61.93752718, 66.7060564 ]
BATCH_SIZE = 512
mean = []
std = []
loaders = {}
for name in ['train', 'test', 'val']:
    if name == 'train':
        mean = train_mean
        std = train_std
    if name == 'val':
        mean = val_mean
        std = val_std
    if name == 'test':
        mean = test_mean
        std = test_std
    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), Squeeze()]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    # Add image transforms and normalization
    if name == 'train':
        image_pipeline.extend([
            RandomHorizontalFlip(),
            RandomTranslate(padding=2),
            Cutout(8, tuple(map(int, mean))), # Note Cutout is done before normalization.
        ])
    image_pipeline.extend([
        ToTensor(),
        #ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(torch.float32),
        torchvision.transforms.Normalize(mean, std)
    ])

    # Create loaders
    loaders[name] = Loader(f'./beton/cifar_{name}_{seed}.beton',
                        batch_size=BATCH_SIZE,
                        num_workers=8,
                        order=OrderOption.RANDOM,
                        drop_last=(name == 'train'),
                        pipelines={'image': image_pipeline,'label': label_pipeline})  # 🔧 Required for GPU transforms
    
print(loaders['val'])


from sklearn.decomposition import PCA

def get_flat_params(model):
    return torch.cat([p.data.flatten().cpu() for p in model.parameters() if p.requires_grad])

def set_flat_params(model, flat_params):
    idx = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(flat_params[idx:idx + numel].view_as(p))
            idx += numel

def plot_loss_contours_and_trajectory(weight_snapshots, net, loss_fn, testloader, steps=40, span=1.0):
    device = next(net.parameters()).device

    # Stack weight snapshots (num_epochs x num_params)
    weights = torch.stack(weight_snapshots)
    weights_np = weights.numpy()

    # PCA to reduce to 2D
    pca = PCA(n_components=2)
    projected = pca.fit_transform(weights_np)

    # Directions in weight space corresponding to PCA components
    pc1 = torch.tensor(pca.components_[0], dtype=torch.float32)
    pc2 = torch.tensor(pca.components_[1], dtype=torch.float32)
    center = weights[-1]  # Final weights as center

    # Create grid around final point in PCA space
    alphas = np.linspace(-span, span, steps)
    betas = np.linspace(-span, span, steps)
    losses = np.zeros((steps, steps))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            direction = center + alpha * pc1 + beta * pc2
            set_flat_params(net, direction.clone())

            net.eval()
            total_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)
                    total_samples += inputs.size(0)
            losses[j, i] = total_loss / total_samples  # note: Y, X indexing

    # Plot contours
    X, Y = np.meshgrid(alphas, betas)
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, losses, levels=30, cmap='viridis')
    plt.colorbar(contour, label="Loss")

    # Plot trajectory arrows
    traj = projected
    plt.plot(traj[:, 0] - traj[-1, 0], traj[:, 1] - traj[-1, 1], 'w.-', label="Trajectory")
    for i in range(len(traj) - 1):
        plt.arrow(
            traj[i, 0] - traj[-1, 0], traj[i, 1] - traj[-1, 1],
            traj[i+1, 0] - traj[i, 0], traj[i+1, 1] - traj[i, 1],
            color='white', alpha=0.8, head_width=0.02, length_includes_head=True
        )

    plt.title("Loss Contours with Training Trajectory")
    plt.xlabel("PCA Direction 1")
    plt.ylabel("PCA Direction 2")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_loss_landscape(model, loss_fn, dataloader, num_points=20, alpha=1.0):
    # Store original parameters
    original_params = [p.clone() for p in model.parameters()]
    
    # Calculate two random directions
    direction1 = [torch.randn_like(p) for p in model.parameters()]
    direction2 = [torch.randn_like(p) for p in model.parameters()]
    
    # Normalize directions
    norm1 = torch.sqrt(sum(torch.sum(d**2) for d in direction1))
    norm2 = torch.sqrt(sum(torch.sum(d**2) for d in direction2))
    direction1 = [d / norm1 for d in direction1]
    direction2 = [d / norm2 for d in direction2]
    
    # Create grid
    x = np.linspace(-alpha, alpha, num_points)  
    y = np.linspace(-alpha, alpha, num_points)
    X, Y = np.meshgrid(x, y)
    
    # Calculate loss for each point
    Z = np.zeros_like(X)
    for i in range(num_points):
        for j in range(num_points):
            # Update model parameters
            for p, d1, d2 in zip(model.parameters(), direction1, direction2):
                p.data = p.data + X[i,j] * d1 + Y[i,j] * d2
            
            # Calculate loss
            total_loss = 0
            num_batches = 0
            for batch in dataloader:
                
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
            Z[i,j] = total_loss / num_batches
            
            # Reset model parameters
            for p, orig_p in zip(model.parameters(), original_params):
                p.data = orig_p.clone()
    
    # Plot the loss landscape
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape')
    fig.colorbar(surf)


    
    plt.tight_layout()
    
    plt.show()

def log_loss_landscape(model, loss_fn, dataloader, step):
    # Generate the loss landscape plot
    plot_loss_landscape(model, loss_fn, dataloader)

from io import BytesIO
def stitch_images_grid(imgs, rows=3, cols=1):
    assert len(imgs) == rows * cols
    widths, heights = zip(*(img.size for img in imgs))
    max_width = max(widths)
    max_height = max(heights)

    stitched_img = Image.new('RGB', (cols * max_width, rows * max_height), color='white')
    
    for idx, img in enumerate(imgs):
        row = idx // cols
        col = idx % cols
        stitched_img.paste(img, (col * max_width, row * max_height))

    return stitched_img

def hidden_weights(net, testloader, device):
    x2_epoch_hidden = []
    x2_rel_epoch_hidden = []
    x3_epoch_hidden = []
    x3_rel_epoch_hidden = []

    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            hidden_x2 = net(inputs, return_lr2=True)
            hidden_x2_rel = net(inputs, return_lr2_rel2=True)
            hidden_x3 = net(inputs, return_lr3=True)
            hidden_x3_rel = net(inputs, return_lr3_rel3=True)
            
            x2_epoch_hidden.append(hidden_x2.cpu())
            x2_rel_epoch_hidden.append(hidden_x2_rel.cpu())
            x3_epoch_hidden.append(hidden_x3.cpu())
            x3_rel_epoch_hidden.append(hidden_x3_rel.cpu())

    return [x2_epoch_hidden,x2_rel_epoch_hidden,x3_epoch_hidden,x3_rel_epoch_hidden]

def cov_matrix(hidden_weights, save):
    cov_list = []

    print(len(hidden_weights[1]))
    length = len(hidden_weights)
    counter=0
    ######initial######
    for i in range(length):
        print("run")
        # save network parameters, losses and accuracy
        X2 = torch.cat(hidden_weights[i][0], dim=0).float() # shape: (n, 10)
        X2_rel = torch.cat(hidden_weights[i][1], dim=0).float() 
        X3 = torch.cat(hidden_weights[i][2], dim=0).float() 
        X3_rel = torch.cat(hidden_weights[i][3], dim=0).float() 
        
        # Center
        X2_centered = (X2 - (X2.mean(dim=0, keepdim=True)))
        X2_rel_centered = (X2_rel - (X2_rel.mean(dim=0, keepdim=True)))
        X3_centered = (X3 - (X3.mean(dim=0, keepdim=True)))
        X3_rel_centered = (X3_rel - (X3_rel.mean(dim=0, keepdim=True)))

        # Covariance between X and Y
        x2_cov_matrix = (X2_centered.T @ X2_centered) / (X2.shape[0] - 1)
        x2_rel_cov_matrix = (X2_rel_centered.T @ X2_rel_centered) / (X2_rel.shape[0] - 1)
        x3_cov_matrix = (X3_centered.T @ X3_centered) / (X3.shape[0] - 1)
        x3_rel_cov_matrix = (X3_rel_centered.T @ X3_rel_centered) / (X3_rel.shape[0] - 1)
        
        # Suppose we reshape a 2D matrix into 3D just to simulate slices
        if i == 0:
            name = "initial"
        elif i == 1:
            name = "middle"
        elif i == 2:
            name = "final"  
        cov_list.append((x2_cov_matrix.cpu().numpy(), f"x2_cov_matrix_{name}"))  # convert to numpy
        cov_list.append((x2_rel_cov_matrix.cpu().numpy(), f"x2_rel_cov_matrix_{name}"))  # convert to numpy
        cov_list.append((x3_cov_matrix.cpu().numpy(), f"x3_cov_matrix_{name}"))  # convert to numpy
        cov_list.append((x3_rel_cov_matrix.cpu().numpy(), f"x3_rel_cov_matrix_{name}"))  # convert to numpy

     
    
    ##########
    

    # Create 1x4 subplots
    fig, axes = plt.subplots(3, 4, figsize=(24, 24))  # Wider layout for better spacing
    axes = axes.flatten()

    # Loop through covariance matrices and axes
    for index, (ax, (matrix, name)) in enumerate(zip(axes, cov_list)):
        # Plot heatmap
        sns.heatmap(matrix,
                    square=True,
                    cbar=True,
                    xticklabels=False,
                    yticklabels=False,
                    cmap='coolwarm',
                    center=0,
                    ax=ax,
                    cbar_kws={'shrink': 0.7, 'label': 'Covariance Value'})

        # Save to Excel
        feature_names = [f"Feature_{i}" for i in range(matrix.shape[0])]
        df_cov = pd.DataFrame(matrix, index=feature_names, columns=feature_names)

        if save:
            df_cov.to_excel(f"/scratch/tcrv4423/covariance/{name}.xlsx")

        # Set title
        ax.set_title(name, pad=10)

    plt.tight_layout()
    plt.show()

# Note that statistics are wrt to uin8 range, [0,255].
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
torch.use_deterministic_algorithms(False)
dataset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=False)
data = np.stack([np.array(img) for img, _ in dataset])  # shape: (50000, 32, 32, 3)

# Compute mean and std per channel
mean = np.array(data.mean(axis=(0, 1, 2)))   # shape: (3,)
std = np.array(data.std(axis=(0, 1, 2)) )     # shape: (3,)

print("Mean:", mean)
print("Std:", std)
print(len(mean))
# Mean: [125.30691805 122.95039414 113.86538318]
# Std: [62.99321928 62.08870764 66.70489964]

BATCH_SIZE = 512

loaders = {}
for name in ['train', 'test', 'val']:
    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    # Add image transforms and normalization
    if name == 'train':
        image_pipeline.extend([
            RandomHorizontalFlip(),
            RandomTranslate(padding=2),
            Cutout(8, tuple(map(int, mean))), # Note Cutout is done before normalization.
        ])

    image_pipeline.extend([
        ToTensor(),
        ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(torch.float32),
        torchvision.transforms.Normalize(mean, std)
    ])

    # Create loaders
    loaders[name] = Loader(f'./beton/cifar_{name}_{seed}.beton',
                        batch_size=BATCH_SIZE,
                        num_workers=8,
                        order=OrderOption.RANDOM,
                        seed=seed,
                        drop_last=(name == 'train'),
                        pipelines={'image': image_pipeline,'label': label_pipeline},
                        )  # 🔧 Required for GPU transforms
    
print(loaders['val'])

class BPLinear(torch.nn.Linear):
    """BP Linear layer"""

    def __str__(self):
        return "BPLinear"
    
class BPConv2d(torch.nn.Conv2d):
    """BP Conv2d layer"""

    def __str__(self):
        return "BPConv2d"
    
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = model  # The original nn.Sequential model

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs  # List of intermediate activations
    
class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * (self.weight)
    
class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class Flatten(torch.nn.Module):
    def forward(self, x): return x.flatten(1) 

class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_out,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         groups=groups, bias=False),
            torch.nn.BatchNorm2d(channels_out, eps=1e-5, momentum=0.1),
            torch.nn.ReLU(inplace=True)
    )



def deconv_bn(channels_in, channels_out, dlr, kernel_size=3, stride=1, padding=1, groups=1):
    return torch.nn.Sequential(
            DecorConv2d(torch.nn.Conv2d, channels_in, channels_out,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         groups=groups, bias=False,  decor_lr=dlr),
            torch.nn.BatchNorm2d(channels_out, eps=1e-5, momentum=0.1),
            torch.nn.ReLU(inplace=True)
    )


class DecorResNet9(torch.nn.Module):
    def __init__(self, in_channels, num_classes, dlr):
        super().__init__()
        
        self.conv1 = deconv_bn(in_channels, 64, dlr, kernel_size=3, stride=1, padding=1)
        #self.pre = deconv_bn(64, 64, dlr, kernel_size=3, stride=1, padding=1)
        self.conv2 = deconv_bn(64, 128, dlr, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2)

        self.res1 = (torch.nn.Sequential(deconv_bn(128, 128, dlr), deconv_bn(128, 128, dlr)))

        self.conv3 = deconv_bn(128, 256, dlr, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv4 = deconv_bn(256, 512, dlr,  kernel_size=3, stride=1, padding=0)
        self.pool3 = torch.nn.MaxPool2d(2)

        self.res3 = (torch.nn.Sequential(deconv_bn(512, 512, dlr), deconv_bn(512, 512, dlr)))
        #self.conv5 = conv_bn(512, 256, kernel_size=1, stride=5, padding=0)
        self.adaptive = torch.nn.AdaptiveMaxPool2d((1, 1))
        #self.classifier = nn.Sequential(Flatten(), torch.nn.Dropout(0.2), torch.nn.Linear(512, num_classes, bias=False), Mul(0.2))
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        DecorLinear(BPLinear, 512, num_classes, dlr),
                                        )

        # self.res2 = Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256)))
        # #self.conv5 = conv_bn(256, 128, kernel_size=3, stride=1, padding=0)
        # self.classifier2 = nn.Sequential(Flatten(), torch.nn.Dropout(0.2), torch.nn.Linear(128, num_classes, bias=False), Mul(0.2))

        
    def forward(self, xb):
        out = self.conv1(xb)
        # out = self.pre(out)
        out = self.conv2(out)
        out = self.pool1(out)

        out = self.res1(out)+out

        out = self.conv3(out)
        out = self.pool2(out)
        out = self.conv4(out)
        out = self.pool3(out)
        
        out = self.res3(out)+out
        #out = self.conv5(out)

        #out = self.adaptive(out)
        out = self.classifier(out)
        return out
    
class ResNet9(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_bn(in_channels, 64, kernel_size=3, stride=1, padding=1)
        #self.pre = conv_bn(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_bn(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2)

        self.res1 = (torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128)))

        self.conv3 = conv_bn(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv4 = conv_bn(256, 512, kernel_size=3, stride=1, padding=0)
        self.pool3 = torch.nn.MaxPool2d(2)

        self.res3 = (torch.nn.Sequential(conv_bn(512, 512), conv_bn(512, 512)))
        #self.conv5 = conv_bn(512, 256, kernel_size=1, stride=5, padding=0)
        self.adaptive = torch.nn.AdaptiveMaxPool2d((1, 1))
        #self.classifier = nn.Sequential(Flatten(), torch.nn.Dropout(0.2), torch.nn.Linear(512, num_classes, bias=False), Mul(0.2))
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes),
                                        )

        # self.res2 = Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256)))
        # #self.conv5 = conv_bn(256, 128, kernel_size=3, stride=1, padding=0)
        # self.classifier2 = nn.Sequential(Flatten(), torch.nn.Dropout(0.2), torch.nn.Linear(128, num_classes, bias=False), Mul(0.2))

        
    def forward(self, xb):
        out = self.conv1(xb)
       # out = self.pre(out)
        out = self.conv2(out)
        out = self.pool1(out)

        out = self.res1(out)+out

        out = self.conv3(out)
        out = self.pool2(out)
        out = self.conv4(out)
        out = self.pool3(out)
        
        out = self.res3(out)+out
        #out = self.conv5(out)

        #out = self.adaptive(out)
        out = self.classifier(out)
        return out

NUM_CLASSES = 10
model = torch.nn.Sequential(
    conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
    conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
    Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
    conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
    torch.nn.MaxPool2d(2),
    Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
    conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
    torch.nn.AdaptiveMaxPool2d((1, 1)),
    Flatten(),
    torch.nn.Linear(128, NUM_CLASSES, bias=False),
    Mul(0.2)
)
model = ResNet9(3, 10)
model = model.to(memory_format=torch.channels_last).cuda()

def train(name, valloader, testloader, trainloader, learning_rate, decor_learning_rate, decay, EG, bias, cov, lossPlot, epoch_num, save):
    statsrec = np.zeros((4,epoch_num))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    project = args.wandb
    wandb.init(
    project= f"{project}",
    name=name, 
    config={
        "learning_rate": f"{learning_rate:.0e}",
        "decor learning rate": f"{decor_learning_rate:.0e}",
        "weight decay": f"{decay:.0e}",
        "epochs": epoch_num,
        "batch size": BATCH_SIZE
    })

    EPOCHS = epoch_num

    model_output = []
    if(decor_learning_rate == 0):
        print("using normal resnet")
        model = ResNet9(3, 10)
        model = model.to(memory_format=torch.channels_last).cuda() 
  
    else:
        print("using decor restnet")
        model = DecorResNet9(3,10,dlr)
        model = model.to(memory_format=torch.channels_last).cuda() 
        if(bias):
            init_tools.set_split_bias(model)
    opt = torch.optim.AdamW(model.parameters(),lr=learning_rate, weight_decay=decay)
    if(EG == "false"):
        opt = torch.optim.AdamW(model.parameters(),lr=learning_rate, weight_decay=decay)
    elif(EG == "true"):
        opt = adamw_eg.AdamWeg(model.parameters(), lr=learning_rate, weight_decay = decay)
    results_path = f"/scratch/tcrv4423/{folder}/results/{seed}/{name}.pt"
    #iters_per_epoch = 40000 // BATCH_SIZE
    # lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
    #                         [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
    #                         [0, 1, 0])
    # scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, 0.01, epochs=EPOCHS, 
    #                                                steps_per_epoch=len(trainloader))
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=epoch_num)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)

    from tqdm import tqdm
    print("training")
    model_output.append(FeatureExtractor(model).cuda())
    for ep in range(EPOCHS):
        
        val_loss = 0.0 
        running_loss = 0
        total = 0
        correct = 0
        
        print("epoch: ",ep)
        model.train()
        n = 0
        for ims, labs in tqdm(trainloader):
            opt.zero_grad(set_to_none=True)
            #with autocast():
                #ims= ims.to(device, memory_format=torch.channels_last, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            out = model(ims)
            loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            running_loss += loss.item()
            n+=1
            scaler.unscale_(opt)

            #Apply gradient clipping
           #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.12)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            _, predicted = torch.max(out.data, 1)
            total += ims.shape[0]    # add in the number of labels in this minibatch
            correct += (predicted==labs).sum().item()  # add in the number of correct labels
  
        ltrn = (running_loss/n)/100
        atrn = correct/total 
        n_val = 0
            

        model.eval()
        val_loss = 0.0
        correct = 0.0
        total = 0.0
        n_val = 0
        with torch.no_grad():
            for ims, labs in tqdm(valloader):
                #with autocast():
                    #ims, labs = ims.to(device), labs.to(device)
                out = model(ims) # Test-time augmentation
                #total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                loss = loss_fn(out, labs)  # Compute loss
                val_loss += loss.item()
                n_val += 1


                _, predicted = torch.max(out.data, 1)
                total += labs.size(0)    # add in the number of labels in this minibatch
                correct += (predicted == labs).sum().item()  # add in the number of correct labels

            aval = round(correct/total, 3)
            lval = val_loss / n_val

            #print(f'Accuracy: {total_correct / total_num * 100:.1f}%')
        statsrec[:,ep] = (ltrn, atrn, lval, aval)
        wandb.log({
            "train_loss": ltrn,
            "train_accuracy": atrn,
            "val_loss": lval,
            "val_accuracy": aval
        }, step=ep)
        print(
            f"Epoch {ep+1}/{epoch_num} | "
            f"Train Loss: {ltrn:.3f} | Train Acc: {atrn:.1%} | "
            f"Validation Loss: {lval:.3f} | " 
            f"Validation Acc: {aval:.1%}% | " 

        )
        if(ep == EPOCHS-1):
            model_output.append(FeatureExtractor(model).cuda())
        

            
    if(save):
        torch.save({"stats": statsrec}, results_path)
        
dataset = args.dataset
lr = args.lr
dlr = args.dlr
decay = args.decay
out_class = 10
if(dlr == 0):
    if(dataset == "10"):
        name = f"cor-10{lr:.0e},{dlr:.0e},{decay:.0e},{seed}"
        training_loader = loaders['test']
        val_loader = loaders['val']
        test_loader = loaders['test']
    elif(dataset == "100"):
        name = f"cor-100{lr:.0e},{dlr:.0e},{decay:.0e},{seed}"
        training_loader = loaders['test']
        val_loader = loaders['val']
        test_loader = loaders['test']

else:
    if(dataset == "10"):
        name = f"decor-10{lr:.0e},{dlr:.0e},{decay:.0e},{seed}"
        training_loader = loaders['test']
        val_loader = loaders['val']
        test_loader = loaders['test']
    elif(dataset == "100"):
        name = f"decor-100{lr:.0e},{dlr:.0e},{decay:.0e},{seed}"
        training_loader = loaders['test']
        val_loader = loaders['val']
        test_loader = loaders['test']

EG = args.EG
#args.EG
splitbias = args.bias
#args.bias
cov = False
lossPlot = False
epoch = 50
save = True

train(name, loaders['val'], loaders['test'], loaders['train'], lr, dlr, decay, EG, splitbias, cov, lossPlot, epoch, save)
