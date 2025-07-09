import torch
import torch.nn as nn

# This function allows various methods to generate adversarial samples.
def base_attack(model, images, labels, epsilon=None,ll=False,alpha=None,rand=False,iters=1):
    images = images.clone().detach()
    ori_images = images.clone().detach()
    images = images.cuda()
    model = model.cuda()

    # When training we can used a fixed epsilon or sample it from a normal distribution
    if epsilon is None:
        eps = 8/255
        # eps = abs(np.random.normal(0, 8, 1)[0])/255
    else:
        eps = epsilon
    
    if iters == 1:
        alpha = eps
    
     # Rand allows to enable R+FGSM, which adds noise to the images before calculating the gradient
    if rand:
        noise = torch.randn_like(images)
        images = images + (eps / 2) * noise    

    for _ in range(iters):
        images.requires_grad = True
    
        output = model(images)
        
        # If ll is true, we use the least likely class as target, otherwise we use the labels
        if ll:
            y = torch.argmin(output, dim=1)
        else:
            y = labels    
        
        # Compute the gradient w.r.t. to the target
        model.zero_grad()
        loss = nn.CrossEntropyLoss()(output, y).to("cuda")
        loss.backward()
        images = images + alpha * images.grad.sign()
        perturbation = torch.clamp(images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + perturbation, min=0, max=1).detach()
    return images

# Simplest attack that adds a small perturbation to the images in the direction of the gradient of the loss w.r.t. the input.
def fgsm_attack(model, images, labels, epsilon=None, ll=False):
    return base_attack(model, images, labels, epsilon=epsilon,alpha=None, ll=ll, rand=False, iters=1)

# Performs the same attack as FGSM but adds noise to the images before calculating the gradient.
def r_fgsm_attack(model, images, labels, epsilon=None,ll=False):
    return base_attack(model, images, labels, epsilon=epsilon,alpha=None, ll=ll, rand=True, iters=1)

# Performs the PGD attack, which is an iterative version of FGSM.
def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, ll=False):
    return base_attack(model, images, labels, epsilon=eps, ll=ll,alpha=alpha, rand=False, iters=iters)

# Performs the PGD attack with random noise added to the images before calculating the gradient.
def r_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, ll=False):
    return base_attack(model, images, labels, epsilon=eps, ll=ll,alpha= alpha, rand=True, iters=iters)