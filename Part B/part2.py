import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import wandb as wb
import gc
import numpy as np
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 32
# Load iNaturalist12k dataset
dataset = torchvision.datasets.ImageFolder(root='/kaggle/input/inaturalist12k/Data/inaturalist_12K/train',transform=transform)
val_size = int(0.2 * len(dataset))
train_size = len(dataset)-val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
test_dataset = torchvision.datasets.ImageFolder(root='/kaggle/input/inaturalist12k/Data/inaturalist_12K/val',transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('Amphibia', 'Animalia', 'Arachnida', 'Aves','Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia')