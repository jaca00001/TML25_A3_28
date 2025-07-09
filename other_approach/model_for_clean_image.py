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

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Hyperparameters
batch_size = 128
model_name = "clean_model.pt"
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
        if self.transform:
            img = self.transform(img)
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
# Load your dataset (replace with actual loading logic)
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

train_loader = DataLoader(MyDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(MyDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

# ------------------ Training ------------------
model = ResNet18Classifier10().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
best_val_acc = 0.0

for epoch in range(100):
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
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

    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    # Save the best model based on validation accuracy
    if acc > best_val_acc:
        best_val_acc = acc
        state_dict = model.state_dict()
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        torch.save(clean_state_dict, model_name)
        print(f"Saved new best model with accuracy: {best_val_acc:.2f}%")
        
# Save model (remove 'model.' prefix for clean state_dict)
#state_dict = model.state_dict()
#clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
#torch.save(clean_state_dict, model_name)
#print(f"Model saved to {model_name}")

# ------------------ Testing ------------------
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(model_name, map_location=device))
model.to(device)
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = 100.0 * correct / total
print(f"Test Accuracy: {acc:.2f}%")

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
