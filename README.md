# SimCLR on CIFAR-10

Implementation of **SimCLR (Simple Framework for Contrastive Learning of Visual Representations)** on CIFAR-10 dataset.

## Overview

SimCLR learns visual representations without labels using contrastive learning. This implementation includes:
- ResNet-18 backbone adapted for 32x32 images
- NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)
- Strong data augmentation (random crop, color jitter, grayscale, Gaussian blur)
- t-SNE visualization of learned embeddings
- Linear probe evaluation

## Results

- **Linear probe accuracy:** 85.96% (200 epochs)
- **Random baseline:** 10%
- **Fully supervised ResNet-18:** ~93%

## Requirements
torch
torchvision
numpy
matplotlib
scikit-learn
tqdm


## Usage

Run the notebook cells sequentially. The notebook includes:
- Automatic checkpoint saving every 5 epochs
- Resume training from last checkpoint
- Model download via Google Colab

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch size | 512 |
| Temperature (τ) | 0.5 |
| Learning rate | 1e-3 |
| Warmup epochs | 10 |
| Projection head dim | 128 |

## Files

- `SimCLR_P1.ipynb` - Main training notebook
- `encoder_weights.pth` - Encoder weights for transfer learning
- `best_model.pth` - Full model with projection head
- `training_curve.png` - Loss over epochs
- `tsne.png` - t-SNE visualization

## References

- [SimCLR Paper](https://arxiv.org/abs/2002.05709) - Chen et al., 2020
