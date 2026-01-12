"""
Inference and Qualitative Evaluation Module

This module loads trained FlexibleDAE models from the 'saved_models' directory 
and generates comparative visualization grids. For each MNIST digit (0-9), 
it produces a grid mapping reconstruction quality across varying network 
depths and bottleneck dimensions (k).
"""

import os
import re
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class FlexibleDAE(nn.Module):
    """
    Symmetric Denoising Autoencoder architecture.
    Identical to the training definition to ensure state_dict compatibility.
    """
    def __init__(self, bottleneck_dim, num_layers):
        super(FlexibleDAE, self).__init__()
        dims = [784, 256, 128] 
        enc_layers = []
        current_dim = 784
        for i in range(min(num_layers - 1, 2)):
            enc_layers.append(nn.Linear(current_dim, dims[i+1]))
            enc_layers.append(nn.ReLU())
            current_dim = dims[i+1]
        enc_layers.append(nn.Linear(current_dim, bottleneck_dim))
        enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)
        
        dec_layers = []
        current_dim = bottleneck_dim
        if num_layers > 2:
            dec_layers.extend([nn.Linear(current_dim, 128), nn.ReLU()])
            current_dim = 128
        if num_layers > 1:
            dec_layers.extend([nn.Linear(current_dim, 256), nn.ReLU()])
            current_dim = 256
        dec_layers.append(nn.Linear(current_dim, 784))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

def get_test_samples(dataset):
    """
    Extracts the first occurrence of each digit (0-9) from the dataset.
    """
    samples = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label not in samples:
            samples[label] = i
        if len(samples) == 10:
            break
    return samples

def generate_comparison_grids():
    """
    Orchestrates the loading of models and generation of comparison figures.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "saved_models"
    output_dir = "comparison_grids"
    os.makedirs(output_dir, exist_ok=True)

    # Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x.view(-1))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    samples_to_process = get_test_samples(test_dataset)

    # Metadata extraction from filenames
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    depths = sorted(list(set(int(re.search(r'depth(\d+)', f).group(1)) for f in model_files)))
    ks = sorted(list(set(int(re.search(r'k(\d+)', f).group(1)) for f in model_files)))

    print(f"\n{' VISUALIZATION PIPELINE ':=^50}")

    for digit, sample_idx in sorted(samples_to_process.items()):
        clean_img, _ = test_dataset[sample_idx]
        clean_tensor = clean_img.to(device).unsqueeze(0)
        
        # Apply deterministic noise for fair comparison across grids
        noise_factor = 0.5
        noise = torch.randn_like(clean_tensor) * noise_factor
        noisy_tensor = torch.clamp(clean_tensor + noise, 0, 1)

        fig, axes = plt.subplots(len(depths) + 1, len(ks), 
                                 figsize=(2.5 * len(ks), 2.5 * (len(depths) + 1)))

        # Row 0: Reference visuals (Original and Noisy)
        axes[0, 0].imshow(clean_tensor.cpu().view(28, 28), cmap='gray')
        axes[0, 0].set_title("ORIGINAL", fontweight='bold', color='#2e7d32', fontsize=10)
        
        if len(ks) > 1:
            axes[0, 1].imshow(noisy_tensor.cpu().view(28, 28), cmap='gray')
            axes[0, 1].set_title("NOISY INPUT", fontweight='bold', color='#c62828', fontsize=10)

        for c in range(len(ks)):
            axes[0, c].axis('off')

        # Rows 1-N: Model Reconstructions
        for r_idx, d in enumerate(depths):
            r = r_idx + 1 
            for c, k in enumerate(ks):
                ax = axes[r, c]
                model_path = os.path.join(model_dir, f"model_depth{d}_k{k}.pth")
                
                if os.path.exists(model_path):
                    model = FlexibleDAE(k, d).to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    
                    with torch.no_grad():
                        restored = model(noisy_tensor).cpu().view(28, 28)
                    
                    ax.imshow(restored, cmap='gray')
                    
                    if r == 1:
                        ax.set_title(f"Bottleneck $k={k}$", fontsize=11, fontweight='bold')
                    if c == 0:
                        ax.set_ylabel(f"Depth $D={d}$", fontsize=11, fontweight='bold', labelpad=15)
                        ax.set_yticks([]) 
                else:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center', color='gray')
                
                ax.set_xticks([])
                if c > 0: ax.set_yticks([])

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"grid_digit_{digit}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Generated Figure: Digit {digit} -> {save_path}")

    print(f"{' PIPELINE COMPLETE ':=^50}\n")

if __name__ == "__main__":
    generate_comparison_grids()

