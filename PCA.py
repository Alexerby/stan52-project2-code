"""
Dimensionality Analysis for Autoencoder Bottleneck Configuration

This module utilizes Principal Component Analysis (PCA) via eigendecomposition 
to determine the optimal number of latent features required to preserve 
specific proportions of variance in the MNIST dataset. It serves as a 
pre-processing step for configuring Denoising Autoencoder (DAE) architectures.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def calculate_pca_variance(data):
    """
    Computes the cumulative explained variance ratio using eigendecomposition.
    """
    centered_data = data - torch.mean(data, dim=0)
    
    n_samples = data.size(0)
    cov = torch.mm(centered_data.t(), centered_data) / (n_samples - 1)
    
    eigenvalues, _ = torch.linalg.eigh(cov)
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    
    var_ratio = eigenvalues / torch.sum(eigenvalues)
    return torch.cumsum(var_ratio, dim=0).numpy()

def plot_variance_apa(cumulative_variance):
    """
    Generates a grayscale, APA-compliant visualization of variance retention.
    """
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    
    plt.plot(cumulative_variance, color='black', linewidth=1.2, label='Cumulative Variance')
    plt.axhline(y=0.9, color='black', linestyle='--', linewidth=0.8, label='90% Threshold')
    plt.axhline(y=0.8, color='black', linestyle=':', linewidth=0.8, label='80% Threshold')

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_facecolor('white')
    
    plt.xlabel('Number of Principal Components', fontsize=10)
    plt.ylabel('Proportion of Variance Explained', fontsize=10)
    plt.legend(frameon=False, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=True)
    images, _ = next(iter(loader))

    cum_var = calculate_pca_variance(images)
    
    print(f"\n{' DIMENSIONALITY ANALYSIS ':=^40}")
    print(f"{'Target Variance':<20} | {'Required k':<15}")
    print("-" * 40)
    
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        k = (cum_var >= threshold).argmax() + 1
        print(f"{threshold:>18.0%} | {k:<15}")
        
    print("-" * 40)
    print(f"Total Input Features: {images.size(1)}")
    print("=" * 40 + "\n")

    plot_variance_apa(cum_var)

if __name__ == "__main__":
    main()
