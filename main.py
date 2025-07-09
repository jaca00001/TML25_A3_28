import torch
import torch.nn as nn
import os
from torchvision import models
import torch
from torch.utils.data import  DataLoader,random_split
from torchvision import transforms

from src.utils import * 
from src.dataset import *

# Enable benchmark mode for faster training
torch.backends.cudnn.benchmark = True

os.makedirs("out/models", exist_ok=True)
os.makedirs("out/plots", exist_ok=True)

if __name__ == "__main__":

    EPOCHS = 10
    BATCH_SIZE = 256
    
    # Turn the images into tensors and map into the range [0, 1]
    transform = transforms.Compose([
      transforms.ToTensor(),
    ])

    # Load dataset, apply RGB here to reduce time spent in training
    dataset = torch.load("data/Train.pt", weights_only=False)
    dataset.transform = transform
    dataset.imgs = [img.convert('RGB') for img in  dataset.imgs]

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader =  DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # Load the base model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.weight.shape[1], 10)
    model_type = "resnet18"

    # Each epoch we record the training loss and test accuracy for epsilon = 0
    train_loss = []
    test_accuracy = []
    for epoch in tqdm(range(EPOCHS)):
        train_loss.append(train(model, train_loader, adv=True))
        test_accuracy.append(evaluate(model, test_loader, epsilon=0))
    torch.save(model.state_dict(), "out/models/adv_model.pt")
    plot_loss(train_loss, test_accuracy, filename="out/plots/adv_loss.png")

    # Upload the model to the server
    upload("out/models/adv_model_top.pt", model_type)
        
