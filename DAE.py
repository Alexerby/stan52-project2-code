"""
Denoising Autoencoder (DAE) Grid Search Module

This module performs an informed architectural search for a Denoising Autoencoder. 
It evaluates different network depths and bottleneck dimensions—derived from 
prior Principal Component Analysis—to identify the optimal balance between 
feature compression and reconstruction fidelity on the MNIST dataset.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Configuration and Hardware Acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("saved_models", exist_ok=True)

# Data Pipeline
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: x.view(-1))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

class FlexibleDAE(nn.Module):
    """
    Symmetric Denoising Autoencoder with configurable depth and bottleneck.
    """
    def __init__(self, bottleneck_dim, num_layers):
        super(FlexibleDAE, self).__init__()
        dims = [784, 256, 128] 
        
        # Encoder Construction
        enc_layers = []
        current_dim = 784
        for i in range(min(num_layers - 1, 2)):
            enc_layers.append(nn.Linear(current_dim, dims[i+1]))
            enc_layers.append(nn.ReLU())
            current_dim = dims[i+1]
        
        enc_layers.append(nn.Linear(current_dim, bottleneck_dim))
        enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder Construction
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
        """
        Maps noisy input to reconstructed clean output.
        """
        return self.decoder(self.encoder(x))

def train_model(b_dim, depth, epochs=50):
    """
    Executes training loop for a specific architecture configuration.
    """
    model_name = f"model_depth{depth}_k{b_dim}.pth"
    model_path = os.path.join("saved_models", model_name)
    
    model = FlexibleDAE(b_dim, depth).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    model.train()
    # Through the entire dataset we use stochastic gradient descent (SGD)
    # and view the data in 81 mini-batches. 
    # Each mini-batch is of size 128
    for epoch in range(epochs): # through the entire dataset
        for batch_idx, (data, _) in enumerate(train_loader):
            if batch_idx > 80: # Limit mini-batches to 81 per epoch
                break 
            
            clean = data.to(device)
            # Add Gaussian noise and clip to valid pixel range [0, 1]
            noisy = torch.clamp(clean + torch.randn_like(clean) * 0.5, 0, 1) # corrupt the data
            
            optimizer.zero_grad()
            output = model(noisy) # forward pass the data 
            loss = criterion(output, clean) # calculate the loss
            loss.backward() # use the chain rule for backpropagation
            optimizer.step() # nudge the weights in the opposite direction of the gradient
            
    torch.save(model.state_dict(), model_path)
    return loss.item()

def main():
    # Experimental Parameters (k values based on 70%, 80%, and 90% PCA variance)
    pca_bottlenecks = [26, 43, 86]
    depth_options = [1, 2, 3]
    grid_results = []

    print(f"\n{' GRID SEARCH INITIALIZED ':=^50}")
    print(f"Compute Device: {device}")

    for d in depth_options:
        for b in pca_bottlenecks:
            print(f"Testing Config: Depth {d} | Bottleneck {b}...")
            final_loss = train_model(b, d)
            
            # Map bottleneck to its variance significance for reporting
            var_map = {26: "70%", 43: "80%", 86: "90%"}
            
            grid_results.append({
                "Depth": d, 
                "Bottleneck": b, 
                "Variance": var_map.get(b),
                "Final_MSE": round(final_loss, 5)
            })

    # Results Tabulation
    results_df = pd.DataFrame(grid_results)
    print(f"\n{' GRID SEARCH COMPLETE ':=^50}")
    
    # Pivot for clearer comparison of Loss vs. Bottleneck Size
    summary_table = results_df.pivot(index="Depth", columns="Bottleneck", values="Final_MSE")
    print("\nMean Squared Error Summary:")
    print(summary_table)
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
