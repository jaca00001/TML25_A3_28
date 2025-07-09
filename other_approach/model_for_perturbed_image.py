import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
from sklearn.model_selection import train_test_split
import requests
from torchattacks import FGSM, PGD
from itertools import cycle

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Hyperparameters
batch_size = 128
trained_model_name = "clean_model.pt"
model_name = "robust_model_v3.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
submit = False

# ------------------ Dataset class ------------------
class TaskDataset(Dataset):
    def __init__(self, transform = None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ------------------ Model ------------------
class ResNet18Classifier10(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

# ------------------ Data Loading ------------------
dataset: TaskDataset = torch.load("Train.pt", weights_only=False)

transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
    transforms.ToTensor(),
])

images = []
labels = []

for i in range(len(dataset)):
    _, image, label = dataset[i]
    image = transform(image)
    images.append(image)
    labels.append(torch.tensor(label).long())


X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.05, random_state=42)

# Load pretrained clean model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(trained_model_name, map_location=device))
model.to(device)
model.eval()

# Generate adversarial images from clean model
fgsm = FGSM(model, eps=0.05)
pgd = PGD(model, eps=0.1, alpha=0.007, steps=14)

print("Generating adversarial training data...")
clean_imgs, clean_lbls = [], []
fgsm_imgs, fgsm_lbls = [], []
pgd_imgs, pgd_lbls = [], []
for images, labels in tqdm(DataLoader(MyDataset(X_train, y_train), batch_size=batch_size)):
    images, labels = images.to(device), labels.to(device)
    fgsm_adv = fgsm(images, labels).detach().cpu()
    pgd_adv  = pgd(images, labels).detach().cpu()
    clean_imgs.extend(images.cpu())
    clean_lbls.extend(labels.cpu())
    fgsm_imgs.extend(fgsm_adv)
    fgsm_lbls.extend(labels.cpu())
    pgd_imgs.extend(pgd_adv)
    pgd_lbls.extend(labels.cpu())

val_clean_imgs, val_clean_lbls = [], []
val_fgsm_imgs, val_fgsm_lbls = [], []
val_pgd_imgs, val_pgd_lbls = [], []
for images, labels in tqdm(DataLoader(MyDataset(X_val, y_val), batch_size=batch_size)):
    images, labels = images.to(device), labels.to(device)
    fgsm_adv = fgsm(images, labels).detach().cpu()
    pgd_adv  = pgd(images, labels).detach().cpu()
    val_clean_imgs.extend(images.cpu())
    val_clean_lbls.extend(labels.cpu())
    val_fgsm_imgs.extend(fgsm_adv)
    val_fgsm_lbls.extend(labels.cpu())
    val_pgd_imgs.extend(pgd_adv)
    val_pgd_lbls.extend(labels.cpu())

# Mixed dataloader
B_CLEAN = batch_size // 2
B_FGSM = batch_size // 4
B_PGD = batch_size // 4

def create_mixed_dataloader():
    clean_loader = DataLoader(MyDataset(clean_imgs, clean_lbls), batch_size=B_CLEAN, shuffle=True, drop_last=True)
    fgsm_loader = DataLoader(MyDataset(fgsm_imgs, fgsm_lbls), batch_size=B_FGSM, shuffle=True, drop_last=True)
    pgd_loader  = DataLoader(MyDataset(pgd_imgs,  pgd_lbls),  batch_size=B_PGD, shuffle=True, drop_last=True)

    def mixed_batches():
        fgsm_iter = cycle(fgsm_loader)
        pgd_iter = cycle(pgd_loader)
        for clean_batch in clean_loader:
            fgsm_batch = next(fgsm_iter)
            pgd_batch  = next(pgd_iter)
            imgs = torch.cat([clean_batch[0], fgsm_batch[0], pgd_batch[0]], dim=0)
            lbls = torch.cat([clean_batch[1], fgsm_batch[1], pgd_batch[1]], dim=0)
            idx = torch.randperm(len(imgs))
            yield imgs[idx], lbls[idx]

    return mixed_batches, len(clean_loader)

def create_val_mixed_dataloader():
    clean_loader = DataLoader(MyDataset(val_clean_imgs, val_clean_lbls), batch_size=B_CLEAN, shuffle=False, drop_last=True)
    fgsm_loader = DataLoader(MyDataset(val_fgsm_imgs, val_fgsm_lbls), batch_size=B_FGSM, shuffle=False, drop_last=True)
    pgd_loader  = DataLoader(MyDataset(val_pgd_imgs,  val_pgd_lbls),  batch_size=B_PGD, shuffle=False, drop_last=True)

    def mixed_batches():
        fgsm_iter = cycle(fgsm_loader)
        pgd_iter = cycle(pgd_loader)
        for clean_batch in clean_loader:
            fgsm_batch = next(fgsm_iter)
            pgd_batch  = next(pgd_iter)
            imgs = torch.cat([clean_batch[0], fgsm_batch[0], pgd_batch[0]], dim=0)
            lbls = torch.cat([clean_batch[1], fgsm_batch[1], pgd_batch[1]], dim=0)
            idx = torch.randperm(len(imgs))
            yield imgs[idx], lbls[idx]

    return mixed_batches, len(clean_loader)

# ------------------ Reinitialize model ------------------
#model = ResNet18Classifier10().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
best_val_acc = 0.0

# ------------------ Training ------------------
for epoch in range(50):
    model.train()
    train_loader_fn, train_batches = create_mixed_dataloader()  # recreate each epoch
    train_loader = train_loader_fn()
    total_loss = 0
    for i, (images, labels) in enumerate(tqdm(train_loader, total=train_batches, desc=f"Training Epoch {epoch+1}")):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch {i+1} : Trainin Loss : {loss.item():.4f}")
        #if i == 100:
        #    break
    print(f"Epoch {epoch+1} Loss: {total_loss / train_batches:.4f}")

    # Validation
    model.eval()
    val_loader_fn, val_batches = create_val_mixed_dataloader()
    val_loader = val_loader_fn()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100.0 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    # Save best model
    if acc > best_val_acc:
        best_val_acc = acc
        state_dict = model.state_dict()
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        torch.save(clean_state_dict, model_name)
        print(f"Saved new best model with accuracy: {best_val_acc:.2f}%")

# ------------------ Final Testing on Validation ------------------
print("\n--- Final Evaluation on Validation Set ---")
allowed_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}
model: torch.nn.Module = allowed_models["resnet50"](weights=None)
model.fc = torch.nn.Linear(model.fc.weight.shape[1], 10)
model.load_state_dict(torch.load(model_name, map_location=device))
model.to(device)
model.eval()
val_loader_fn, val_batches = create_val_mixed_dataloader()
val_loader = val_loader_fn()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in tqdm(val_loader, total=val_batches, desc="Final Validation Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
acc = 100.0 * correct / total
print(f"Final Test Accuracy on Validation Set: {acc:.2f}%")

# ------------------ Submission ------------------
if submit:
    allowed_models = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    with open(model_name, "rb") as f:
        try:
            model: torch.nn.Module = allowed_models["resnet50"](weights=None)
            model.fc = torch.nn.Linear(model.fc.weight.shape[1], 10)
        except Exception as e:
            raise Exception(
                f"Invalid model class, {e=}, only {allowed_models.keys()} are allowed",
            )
        try:
            state_dict = torch.load(f, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            out = model(torch.randn(1, 3, 32, 32))
        except Exception as e:
            raise Exception(f"Invalid model, {e=}")

    token = '08392413'
    response = requests.post(
        "http://34.122.51.94:9090/robustness",
        files={"file": open(model_name, "rb")},
        headers={"token": token, "model-name": "resnet50"}
    )
    print(response.json())
