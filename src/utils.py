import os
import random
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from typing import Tuple
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn

from scipy.stats import truncnorm
def fgsm_attack(model, images, labels, eps=0.3):
    images = images.clone().detach().requires_grad_(True)
    output = model(images)
    loss = nn.CrossEntropyLoss()(output, labels)
    grad = torch.autograd.grad(loss, images)[0]
    perturbed_image = images + eps * grad.sign()
    return torch.clamp(perturbed_image, 0, 1)


def step_LL_attack(model, images, labels, eps=0.3):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    y_ll = outputs.argmin(dim=1)
    loss = nn.CrossEntropyLoss()(outputs, y_ll)
    grad = torch.autograd.grad(loss, images)[0]
    images = images - eps * grad.sign()
    return torch.clamp(images, 0, 1)



def pgd_attack_ll(model, images, labels, eps=0.3, alpha=1):
    device = images.device
    batch_size = images.size(0)

    eps = eps / 256

    # Determine number of iterations based on max epsilon
    eps_max = eps.max().item() if eps.ndim > 0 else eps.item()
    iters = int(np.ceil(min(eps_max + 4, 1.25 * eps_max)))

    # Prepare data
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    eps = eps.view(-1, 1, 1, 1).to(device)  # reshape for broadcasting
    model = model.to(device)

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        y_ll = outputs.argmin(dim=1)
        loss = nn.CrossEntropyLoss()(outputs, y_ll)

        model.zero_grad()
        loss.backward()
        grad = images.grad

        # Update and clip perturbation
        images = images + alpha * grad.sign()
        perturbation = torch.clamp(images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + perturbation, min=0, max=1).detach()

    return images

def pgd_attack(model, images, labels, eps, alpha=1):
    device = images.device
    batch_size = images.size(0)

    eps = eps / 256
    alpha = alpha / 256


    # Determine number of iterations based on max epsilon
    eps_max = eps.max().item() if eps.ndim > 0 else eps.item()
    iters = int(np.ceil(min(eps_max + 4, 1.25 * eps_max)))

    # Prepare data
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    eps = eps.view(-1, 1, 1, 1).to(device)  # reshape for broadcasting
    model = model.to(device)

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()
        grad = images.grad

        # Update and clip perturbation
        images = images + alpha * grad.sign()
        perturbation = torch.clamp(images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + perturbation, min=0, max=1).detach()

    return images


def R_FGSM_attack(model, images, labels, eps):
    device = images.device
    images = images.clone().detach().to(device)
    eps = eps.view(-1, 1, 1, 1).to(device)  # shape [B, 1, 1, 1]
    eps = eps / 256

    # Add random noise
    noise = torch.randn_like(images)
    images_ = images + (eps / 2) * noise
    images_ = images_.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(images_)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    # Compute gradients
    grad = torch.autograd.grad(loss, images_, create_graph=False)[0]

    # Apply FGSM step
    perturbed = images_ + (eps - eps / 2) * grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)

    return perturbed.detach()



# ==== Utilities ====

def trunc_norm(mean=0, std=4, low=0, high=8, size=1):
    mu = mean
    sigma = std
    a_, b_ = (low - mu) / sigma, (high - mu) / sigma
    return truncnorm.rvs(a_, b_, loc=mu, scale=sigma, size=size)


# ==== Training Loop ====

def train(model, loader, test_loader, epochs):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.045)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()

    epoch_combined_losses = []
    epoch_clean_accs = []
    epoch_adv_accs = []

    alpha = 0.2

    for epoch in range(epochs):
        model.train()
        total_combined_loss = 0
        clean_correct = 0
        adv_correct = 0
        count = 0

        for _, imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_size = imgs.size(0)
            ot = batch_size // 3

            perm = torch.randperm(batch_size)
            clean_idx = perm[ot:]
            adv_idx = perm[:ot]

            imgs, labels = imgs.cuda(), labels.cuda()
            clean_imgs = imgs[clean_idx]
            clean_labels = labels[clean_idx]

            clean_pred = model(clean_imgs)
            clean_loss = criterion(clean_pred, clean_labels)


            adv_imgs = imgs[adv_idx]
            adv_labels = labels[adv_idx]
            eps = trunc_norm(mean=0, std=10, low=0, high=20, size=ot) 
            eps = torch.tensor(eps, device=imgs.device).float()

            adv_imgs = R_FGSM_attack(model, adv_imgs, adv_labels, eps).detach()

            adv_pred = model(adv_imgs)
            adv_loss = criterion(adv_pred, adv_labels)

           
            factor = 1 / (ot + alpha * ot)

         
            combined_loss = factor * (clean_loss + alpha * adv_loss)

            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

            total_combined_loss += combined_loss.item() * batch_size
            clean_correct += (clean_pred.argmax(dim=1) == clean_labels).sum().item()
            adv_correct += (adv_pred.argmax(dim=1) == adv_labels).sum().item()
            count += batch_size

        epoch_combined_losses.append(total_combined_loss / count)
        epoch_clean_accs.append(clean_correct / count)
        epoch_adv_accs.append(adv_correct / count)

        test_eps = torch.zeros(64)
        test_acc = evaluate_adversarial(model, test_loader, test_eps)

        print(f"[Epoch {epoch+1}] Combined Loss: {epoch_combined_losses[-1]:.4f} | "
              f"Clean Acc: {epoch_clean_accs[-1]*100:.2f}% | Adv Acc: {epoch_adv_accs[-1]*100:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

        torch.cuda.empty_cache()

    return model


# ==== Evaluation ====

def evaluate_adversarial(model, test_loader, epsilon):
    model.eval()
    correct = 0
    total = 0
    model = model.to("cuda")
    epsilon = epsilon.to("cuda")

    for _, imgs, labels in tqdm(test_loader, leave=False):
        imgs, labels = imgs.cuda(), labels.cuda()

        if epsilon.min() > 0.0:
           
            imgs = pgd_attack(model, imgs, labels, eps=epsilon).detach()

        output = model(imgs)
        pred = output.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    return acc
