# CLVAE: Triplet Loss Variational Autoencoder for Drug Clustering

A PyTorch implementation of a VAE with triplet loss for clustering drug molecules based on molecular descriptors.

## Files

### `CLVAE.py`
Main script containing:
- `TripletCVAE`: VAE model with encoder-decoder architecture
- `TripletVAELoss`: Combined loss (reconstruction + KL + triplet)
- Training and evaluation functions

### `saved_model/triplet_clvae_model.pt`
Pre-trained model weights (22 input features → 3D latent space)

### `all_data_train_set.csv` / `all_data_validation_set.csv`
Training and validation datasets with 22 molecular descriptors + source labels (FDA/ChEMBL/ZINC/GDB17)

### `drug_scalersV2.pkl`
Feature scalers for preprocessing molecular descriptors

## Usage

### Train new model:
```python
from CLVAE import train_triplet_clvae

model, history = train_triplet_clvae(
    data_path="all_data_train_set.csv",
    scaler_path="drug_scalersV2.pkl",
    epochs=100
)
```

### Load pre-trained model:
```python
from CLVAE import TripletCLVAE
import torch

model = TripletCVAE(input_dim=22, latent_dim=3)
model.load_state_dict(torch.load('saved_model/triplet_clvae_model.pt'))
```

## Requirements
```bash
pip install torch numpy pandas scikit-learn
```