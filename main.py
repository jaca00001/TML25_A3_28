import requests
import torch
import torch.nn as nn
import os
from torchvision import models
import torch
from torch.utils.data import  DataLoader,random_split
from torchvision import transforms


import matplotlib.pyplot as plt
import numpy as np

from src.utils import * 

from src.dataset import *


device = torch.device("cuda")
os.makedirs("out/models", exist_ok=True)
os.makedirs("out/plots", exist_ok=True)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    # Enable benchmark mode for faster training
    torch.backends.cudnn.benchmark = True

    model_type = "resnet34"
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.weight.shape[1], 10)

    # Load dataset
    dataset = torch.load("Train.pt", weights_only=False)
    dataset.transform = transform
    dataset.imgs = [img.convert('RGB') for img in  dataset.imgs]

    # Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True,drop_last=True)
    test_loader =  DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True,drop_last=True)


    model = train(model,train_loader,test_loader,10)
    torch.save(model.state_dict(), "out/models/dummy_submission.pt")

    model = models.resnet34(weights=None).to(device)
    model.fc = nn.Linear(model.fc.weight.shape[1], 10)
    model.load_state_dict(torch.load("out/models/dummy_submission.pt", map_location="cuda",weights_only=False))


    for e in np.arange(0,1,0.1):
      eps = torch.ones(128) * e
      print(evaluate_adversarial(model, test_loader,eps))