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


# Enable benchmark mode for faster training
torch.backends.cudnn.benchmark = True

os.makedirs("out/models", exist_ok=True)
os.makedirs("out/plots", exist_ok=True)

if __name__ == "__main__":

    EPOCHS = 20


    transform = transforms.Compose([
      transforms.ToTensor(),
    ])


    # Load dataset
    dataset = torch.load("Train.pt", weights_only=False)
    dataset.transform = transform
    dataset.imgs = [img.convert('RGB') for img in  dataset.imgs]

    # Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader =  DataLoader(test_dataset,  batch_size=256, shuffle=False)


    #load the basemodel
    model_type = "resnet18"

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.weight.shape[1], 10)


    train_loss = []
    test_accuracy = []
    for epoch in tqdm(range(5)):
        train_loss.append(train(model, train_loader, adv=True))
        test_accuracy.append(evaluate(model, test_loader, epsilon=0))
    torch.save(model.state_dict(), "out/models/adv_model.pt")
    plot_loss(train_loss, test_accuracy, filename="out/plots/adv_loss.png")


        
      
