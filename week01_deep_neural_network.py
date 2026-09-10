"""Week 1 - Deep neural networks & training basics (image classification).

Trains a torchvision ResNet-18 on a small FashionMNIST subset, reports test
accuracy, and shows a few predictions vs. ground truth.

This file holds the *model* and the train/evaluate helpers. The five ideas of
week 1 - backpropagation, activation functions + initialisation, dropout,
batch normalisation and residual learning - are each demonstrated with a small
PyTorch example in `week01_deep_neural_network_benchmark.ipynb`, which imports
the helpers below.

Requires: torch, torchvision (CPU is enough; first run downloads ~30 MB of data).
Run: python week01_deep_neural_network.py
"""

# %% [1] Imports and configuration
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
DATA_ROOT = "./data"
TRAIN_SUBSET = 4000
TEST_SUBSET = 1000
BATCH_SIZE = 128
EPOCHS = 2
LR = 1e-3

# Model options (see build_model). "cifar" is the right stem for 28x28 images.
STEM = "cifar"
DROPOUT = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# %% [2] Data (small subset for fast runs)
def build_loaders():
    # ResNet expects 3 channels; replicate the single grayscale channel.
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

    train_full = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_full = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    g = torch.Generator().manual_seed(SEED)
    train_idx = torch.randperm(len(train_full), generator=g)[:TRAIN_SUBSET]
    test_idx = torch.randperm(len(test_full), generator=g)[:TEST_SUBSET]

    train_ds = Subset(train_full, train_idx.tolist())
    test_ds = Subset(test_full, test_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


# %% [3] Model (torchvision ResNet-18, BatchNorm built in)
def build_model(pretrained: bool = False, stem: str = None,
                dropout: float = None) -> nn.Module:
    """Return a ResNet-18 with a 10-class head.

    pretrained : load ImageNet weights instead of training from scratch.
                 (Notebook exercise: compare the first-epoch accuracy.)
    stem       : "cifar"     -> 3x3 stride-1 conv, no max-pool. Right for 28x28.
                 "imagenet"  -> the original 7x7 stride-2 conv + max-pool.
                 Why it matters: with the ImageNet stem a 28x28 image is already
                 7x7 when it reaches layer1, and 1x1 when it reaches layer4, so
                 the deep residual stages have almost no spatial map left to
                 work on. Section 5 of the notebook prints both shape tables.
    dropout    : if > 0, insert nn.Dropout before the classifier.
                 CAREFUL: PyTorch's p is the *drop* probability, while the
                 dropout paper's p is the *keep* probability (lecture 3.4).
                 dropout=0.2 here == the paper's p=0.8.
    """
    stem = STEM if stem is None else stem
    dropout = DROPOUT if dropout is None else dropout

    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.resnet18(weights=weights)

    if stem == "cifar":
        # Replacing conv1 also discards the pretrained stem weights; the rest of
        # the pretrained backbone is kept.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif stem != "imagenet":
        raise ValueError(f"stem must be 'cifar' or 'imagenet', got {stem!r}")

    head = nn.Linear(model.fc.in_features, len(CLASSES))
    model.fc = head if dropout <= 0 else nn.Sequential(nn.Dropout(p=dropout), head)
    return model.to(DEVICE)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def stage_shapes(model: nn.Module, size: int = 28):
    """Spatial shape after the stem and after each residual stage.

    Used by the notebook to show why the ImageNet stem is a poor fit for 28x28.
    """
    was_training = model.training
    model.eval()
    x = torch.zeros(1, 3, size, size, device=next(model.parameters()).device)
    out = {}
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    out["stem"] = tuple(x.shape[1:])
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        x = getattr(model, name)(x)
        out[name] = tuple(x.shape[1:])
    model.train(was_training)
    return out


# %% [4] Train / eval utilities
def train_one_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    running = 0.0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader) -> float:
    model.eval()
    correct = 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        preds = model(images).argmax(dim=1)
        correct += (preds == targets).sum().item()
    return correct / len(loader.dataset)


# %% [5] Training loop
def train(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        acc = evaluate(model, test_loader)
        print(f"epoch {epoch}/{EPOCHS}  train_loss={loss:.4f}  test_acc={acc:.4f}")


# %% [6] Inference on a few samples
@torch.no_grad()
def show_predictions(model, loader, n: int = 8) -> None:
    model.eval()
    images, targets = next(iter(loader))
    images, targets = images[:n].to(DEVICE), targets[:n]
    preds = model(images).argmax(dim=1).cpu()
    print("\nSample predictions (pred vs. true):")
    for i in range(n):
        mark = "OK " if preds[i] == targets[i] else "XX "
        print(f"  {mark} pred={CLASSES[preds[i]]:<12} true={CLASSES[targets[i]]}")


# %% [7] main
def main() -> None:
    set_seed(SEED)
    print(f"device: {DEVICE}")

    train_loader, test_loader = build_loaders()
    model = build_model()
    print(f"stem={STEM}  parameters={count_parameters(model) / 1e6:.2f} M")
    print("feature map per stage (28x28 input):", stage_shapes(model))

    train(model, train_loader, test_loader)
    final_acc = evaluate(model, test_loader)
    print(f"\nfinal test accuracy: {final_acc:.4f}")

    show_predictions(model, test_loader)


if __name__ == "__main__":
    main()
