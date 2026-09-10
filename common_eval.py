"""Shared evaluation helpers for the Advanced Deep Learning notebooks.

This file holds small utilities that more than one weekly notebook needs, so we
do not copy the same code into every notebook.

What is inside:
    Plotting        show_image_grid, plot_curves
    Text metrics    corpus_bleu
    Image metrics   MnistCNN, train_mnist_cnn, cnn_features_and_probs,
                    classifier_score, frechet_distance
    Latent metrics  linear_probe_accuracy, mmd_rbf

Every function is written with plain PyTorch / NumPy so it is easy to read.
Requires: torch, numpy, matplotlib (scipy is optional and only speeds up FID).
"""

from collections import Counter

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def show_image_grid(images, title="", nrow=8, figsize=(8, 8), value_range=None):
    """Show a batch of images as one grid inside a notebook.

    images      : tensor of shape (N, C, H, W)
    value_range : (low, high) of the input pixels, e.g. (-1, 1). If given, the
                  images are rescaled to [0, 1] before drawing.
    """
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    images = images.detach().cpu().float()
    if value_range is not None:
        low, high = value_range
        images = (images - low) / (high - low)
    images = images.clamp(0, 1)

    grid = make_grid(images, nrow=nrow, padding=2)
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_curves(history, keys, title="", xlabel="epoch", ylabel="value",
                figsize=(6, 4)):
    """Plot one line per key from a dict of lists, e.g. {'train': [...], ...}."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    for key in keys:
        values = history[key]
        plt.plot(range(1, len(values) + 1), values, marker="o", label=key)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Text metric: corpus BLEU
# ---------------------------------------------------------------------------


def _ngram_counts(tokens, n):
    """Count all n-grams in one token list."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hypotheses, references, max_n=4):
    """Corpus-level BLEU with one reference per sentence.

    hypotheses : list of token lists produced by the model
    references : list of token lists written by a human
    Returns a score between 0 and 100 (higher is better).

    BLEU = brevity_penalty * exp( mean_n log precision_n ).
    We use the usual "+1 smoothing" so that a zero 4-gram match does not make
    the whole score zero on small test sets.
    """
    clipped = [0] * (max_n + 1)
    total = [0] * (max_n + 1)
    hyp_len, ref_len = 0, 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_counts = _ngram_counts(hyp, n)
            ref_counts = _ngram_counts(ref, n)
            # "Clip" each n-gram count by how often it appears in the reference.
            clipped[n] += sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())
            total[n] += max(0, len(hyp) - n + 1)

    log_precisions = []
    for n in range(1, max_n + 1):
        if total[n] == 0:
            return 0.0
        # +1 smoothing on both numerator and denominator.
        precision = (clipped[n] + 1.0) / (total[n] + 1.0)
        log_precisions.append(math.log(precision))

    # Brevity penalty punishes translations that are too short.
    if hyp_len == 0:
        return 0.0
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100.0 * brevity * math.exp(sum(log_precisions) / max_n)


# ---------------------------------------------------------------------------
# Image-generation metrics: a small MNIST CNN used as the "judge"
# ---------------------------------------------------------------------------


class MnistCNN(nn.Module):
    """Small CNN used to score generated MNIST-like images.

    Real papers use an Inception network trained on ImageNet. For 28x28 digits
    that is overkill, so we train this tiny CNN instead. It gives us
        - class probabilities  -> a "classifier score" (same idea as Inception Score)
        - a 64-dim feature     -> a Frechet distance (same idea as FID)
    """

    def __init__(self, num_classes=10, feature_dim=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 14 -> 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features=False):
        features = self.body(x)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits


def train_mnist_cnn(train_images, train_labels, device, epochs=2, batch_size=256,
                    lr=1e-3, verbose=True):
    """Train the judge CNN on real MNIST images.

    train_images : (N, 1, 28, 28) float tensor with pixels in [0, 1]
    train_labels : (N,) long tensor
    """
    model = MnistCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(train_images, train_labels),
                        batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(1, epochs + 1):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        if verbose:
            print(f"  judge CNN epoch {epoch}/{epochs}  "
                  f"loss={loss_sum / total:.4f}  train_acc={correct / total:.4f}")
    model.eval()
    return model


@torch.no_grad()
def cnn_features_and_probs(model, images, device, batch_size=256):
    """Run the judge CNN and return (features, class probabilities)."""
    model.eval()
    feats, probs = [], []
    for i in range(0, images.size(0), batch_size):
        batch = images[i:i + batch_size].to(device)
        logits, feature = model(batch, return_features=True)
        feats.append(feature.cpu())
        probs.append(logits.softmax(dim=1).cpu())
    return torch.cat(feats), torch.cat(probs)


def classifier_score(probs, splits=1):
    """Inception-Score style number computed with our small MNIST CNN.

    score = exp( mean_x KL( p(y|x) || p(y) ) )
    A high score means: each image looks like one clear digit (low entropy of
    p(y|x)) AND the whole batch covers many digits (high entropy of p(y)).
    The best possible value with 10 classes is 10.0.
    """
    scores = []
    n = probs.size(0)
    for k in range(splits):
        part = probs[k * n // splits:(k + 1) * n // splits]
        marginal = part.mean(dim=0, keepdim=True)
        kl = (part * (part.clamp_min(1e-12).log() - marginal.clamp_min(1e-12).log()))
        scores.append(kl.sum(dim=1).mean().exp().item())
    return float(np.mean(scores))


def class_entropy(probs):
    """Entropy of the average predicted class distribution (mode coverage).

    log(10) = 2.303 is the best value: all ten digits appear equally often.
    A small value is a warning sign of mode collapse.
    """
    marginal = probs.mean(dim=0)
    return float(-(marginal * marginal.clamp_min(1e-12).log()).sum())


def frechet_distance(features_real, features_fake):
    """Frechet distance between two sets of features (the "FID" formula).

    FID = ||mu_r - mu_f||^2 + Tr( C_r + C_f - 2 (C_r C_f)^{1/2} )

    Because we use our small MNIST CNN instead of Inception-v3, the number is
    NOT comparable to published FID values. It is still useful to compare two
    models that were scored with the same judge CNN. Lower is better.
    """
    x = features_real.double().numpy()
    y = features_fake.double().numpy()
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)

    diff = mu_x - mu_y

    def trace_sqrt_product(a, b):
        """Tr( (A B)^(1/2) ), computed in a numerically safe way."""
        try:
            import warnings

            from scipy import linalg

            # A tiny value on the diagonal keeps the matrix invertible; this is
            # the usual trick in public FID implementations.
            eps = 1e-6
            offset = np.eye(a.shape[0]) * eps
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                root, _ = linalg.sqrtm((a + offset).dot(b + offset), disp=False)
            root = np.real(root)
            if np.isfinite(root).all():
                return float(np.trace(root))
        except Exception:
            pass
        # Fallback: the eigenvalues of A B are the squares of what we need.
        eigenvalues = np.linalg.eigvals(a.dot(b))
        return float(np.sum(np.sqrt(np.abs(np.real(eigenvalues)))))

    return float(diff.dot(diff) + np.trace(cov_x) + np.trace(cov_y)
                 - 2 * trace_sqrt_product(cov_x, cov_y))


# ---------------------------------------------------------------------------
# Latent-space metrics
# ---------------------------------------------------------------------------


def linear_probe_accuracy(train_z, train_y, test_z, test_y, device,
                          epochs=200, lr=1e-2, weight_decay=1e-4, verbose=False):
    """Train one linear layer on frozen latent codes and report test accuracy.

    This is the standard "linear probe": if a simple linear model can read the
    class out of z, then the encoder learned a useful representation.
    """
    num_classes = int(max(train_y.max().item(), test_y.max().item())) + 1
    train_z = train_z.to(device).float()
    test_z = test_z.to(device).float()
    train_y = train_y.to(device).long()
    test_y = test_y.to(device).long()

    # Standardise the codes so the probe trains at a stable speed.
    mean, std = train_z.mean(0, keepdim=True), train_z.std(0, keepdim=True) + 1e-6
    train_z = (train_z - mean) / std
    test_z = (test_z - mean) / std

    probe = nn.Linear(train_z.size(1), num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for step in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(probe(train_z), train_y)
        loss.backward()
        optimizer.step()
        if verbose and (step + 1) % 50 == 0:
            print(f"  probe step {step + 1}/{epochs}  loss={loss.item():.4f}")

    with torch.no_grad():
        accuracy = (probe(test_z).argmax(1) == test_y).float().mean().item()
    return accuracy


def mmd_rbf(x, y, sigmas=(0.5, 1.0, 2.0, 4.0)):
    """Maximum Mean Discrepancy between two point clouds (RBF kernel).

    MMD is close to 0 when the two samples look like they come from the same
    distribution. We use it to check how well an encoder's q(z) matches the
    prior p(z) = N(0, I). Several kernel widths are averaged so the result does
    not depend on one lucky choice of sigma.
    """
    x = x.double()
    y = y.double()

    def kernel(a, b):
        d2 = torch.cdist(a, b).pow(2)
        return sum(torch.exp(-d2 / (2 * s ** 2)) for s in sigmas) / len(sigmas)

    n, m = x.size(0), y.size(0)
    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)
    # Remove the diagonal (a point compared with itself) for an unbiased value.
    sum_xx = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1))
    sum_yy = (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1))
    return float(sum_xx + sum_yy - 2 * k_xy.mean())
