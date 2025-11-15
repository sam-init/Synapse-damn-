import os
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "/mnt/d/Web Development/Synapse-damn-/dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")   # used only for final eval

BATCH_SIZE = 16
NUM_EPOCHS = 50          # max epochs; early stopping will cut this short
LR = 1e-4
NUM_WORKERS = 4          # if you get dataloader issues in WSL, set this to 0

PATIENCE = 10            # early stopping patience (epochs without improvement)
MIN_DELTA = 1e-4         # minimum improvement in val acc to reset patience

# -----------------------
# DEVICE - FORCE CUDA
# -----------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is NOT available, but you said 'dont use cpu'. Fix your CUDA/driver/CUDA toolkit setup first.")

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")

# -----------------------
# TRANSFORMS
# -----------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------
# DATASETS & DATALOADERS
# -----------------------
full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
num_classes = len(full_train_dataset.classes)
print("Classes (label order):", full_train_dataset.classes)
# e.g. ['minor', 'moderate', 'severe']

# Split train into train + val
val_ratio = 0.2
n_total = len(full_train_dataset)
n_val = int(n_total * val_ratio)
n_train = n_total - n_val

train_dataset, val_dataset = random_split(full_train_dataset, [n_train, n_val])

# For validation, use deterministic transforms
val_dataset.dataset.transform = val_test_transform

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# Test dataset (for final eval)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# -----------------------
# MODEL (ResNet18 transfer learning)
# -----------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace final FC layer with dropout + linear (regularization)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, num_classes)
)

model = model.to(DEVICE, non_blocking=True)

criterion = nn.CrossEntropyLoss()

# L2 regularization via weight_decay
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# Optional: LR scheduler (helps a bit sometimes)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# -----------------------
# TRAINING / EVAL FUNCTIONS
# -----------------------
def train_one_epoch(epoch, model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Epoch [{epoch}] Step [{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    elapsed = time.time() - start_time
    print(
        f"Train Epoch {epoch}: Loss={epoch_loss:.4f}, "
        f"Acc={epoch_acc:.4f}, Time={elapsed:.1f}s"
    )
    return epoch_loss, epoch_acc


def eval_model(model, loader, criterion, device, mode="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    print(f"{mode}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc

# -----------------------
# TRAINING LOOP + EARLY STOPPING
# -----------------------
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

for epoch in range(1, NUM_EPOCHS + 1):
    print("=" * 60)
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    print("=" * 60)

    train_loss, train_acc = train_one_epoch(
        epoch, model, train_loader, optimizer, criterion, DEVICE
    )
    val_loss, val_acc = eval_model(
        model, val_loader, criterion, DEVICE, mode="Val"
    )

    # Step LR scheduler on val_acc
    scheduler.step(val_acc)

    # Check improvement for early stopping
    if val_acc > best_val_acc + MIN_DELTA:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        print(f" New best Val Acc: {best_val_acc:.4f}")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)")

    if epochs_no_improve >= PATIENCE:
        print("\n Early stopping triggered!")
        break

print(f"\nBest Val Acc achieved: {best_val_acc:.4f}")

# Load best weights before test eval
model.load_state_dict(best_model_wts)

# -----------------------
# FINAL TEST EVALUATION
# -----------------------
print("\nEvaluating best model on TEST set:")
test_loss, test_acc = eval_model(model, test_loader, criterion, DEVICE, mode="Test")
print(f"Test Acc: {test_acc:.4f}")

# -----------------------
# SAVE BEST MODEL
# -----------------------
os.makedirs("weights", exist_ok=True)
model_path = "weights/vehicle_damage_resnet18.pth"
torch.save(model.state_dict(), model_path)
print(f"\nModel saved → {model_path}")
