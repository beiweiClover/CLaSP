import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import pickle
import random
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# Molecular descriptors for drug molecules
FEATURES = ['LogP', 'MolecularWeight', 'HBD', 'HBA', 'ROTB', 'SA_Score', 'F50_prob',
           'PPB_prob', 'CYP3A4_substrate_prob', 'CYP3A4_inhibitor_prob',
           'CYP2D6_substrate_prob', 'CYP2C9_inhibitor_prob', 'CLp_c_prob',
           'FDAMDD_c_prob', 'DILI_prob', 'Micronucleus_prob',
           'Reproductive_toxicity_prob', 'Ames_prob', 'BSEP_inhibitor_prob',
           'Pgp_inhibitor_prob', 'Neurotoxicity_prob', 'VDss']

# Data source mapping
SOURCE_MAPPING = {
    'FDA': 0,      # FDA approved drugs
    'ChEMBL': 1,   # ChEMBL database
    'ZINC': 2,     # ZINC database
    'GDB17': 3     # GDB17 database
}

FDA_CLASS = 0  # FDA approved drugs as positive class


class TripletCLVAE(nn.Module):
    """
    Conditional Variational Autoencoder with Triplet Loss
    
    Architecture:
    - Encoder: Learns latent representations of molecular descriptors
    - Decoder: Reconstructs molecular descriptors from latent space
    - Triplet Loss: Ensures FDA drugs cluster together in latent space
    """
    
    def __init__(self, input_dim, latent_dim, hidden_dims=[128, 64], dropout_rate=0.2):
        super().__init__()
        
        # Encoder network
        self.encoder = nn.Sequential()
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.encoder.add_module(f'linear_{i}', nn.Linear(prev_dim, hidden_dim))
            self.encoder.add_module(f'relu_{i}', nn.ReLU())
            self.encoder.add_module(f'bn_{i}', nn.BatchNorm1d(hidden_dim))
            self.encoder.add_module(f'dropout_{i}', nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder network
        self.decoder = nn.Sequential()
        hidden_dims_rev = hidden_dims[::-1]
        prev_dim = latent_dim
        for i, hidden_dim in enumerate(hidden_dims_rev):
            self.decoder.add_module(f'linear_{i}', nn.Linear(prev_dim, hidden_dim))
            self.decoder.add_module(f'relu_{i}', nn.ReLU())
            self.decoder.add_module(f'bn_{i}', nn.BatchNorm1d(hidden_dim))
            self.decoder.add_module(f'dropout_{i}', nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.decoder.add_module('output', nn.Linear(hidden_dims_rev[-1], input_dim))
    
    def encode(self, x):
        """Encode input to latent space parameters"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space to output"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through the entire network"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var, z


class TripletVAELoss(nn.Module):
    """
    Combined loss function for Triplet CVAE:
    - Reconstruction Loss: MSE between input and reconstruction
    - KL Divergence: Regularizes latent space
    - Triplet Loss: Ensures FDA drugs cluster together
    """
    
    def __init__(self, reconstruction_weight=0.5, kl_weight=1e-3, 
                 triplet_weight=0.5, margin=1.0):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.triplet_weight = triplet_weight
        self.margin = margin
    
    def reconstruction_loss(self, recon_x, x):
        """MSE reconstruction loss"""
        return F.mse_loss(recon_x, x, reduction='mean')
    
    def kl_divergence(self, mu, log_var):
        """KL divergence loss for VAE"""
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    def triplet_loss(self, z, labels, negative_labels=None):
        """
        Triplet loss to cluster FDA drugs together
        
        For each FDA drug (anchor):
        - Find another FDA drug as positive
        - Find non-FDA drug as negative
        - Minimize distance(anchor, positive) - distance(anchor, negative) + margin
        """
        if negative_labels is None:
            negative_labels = [i for i in range(4) if i != FDA_CLASS]
        
        fda_indices = (labels == FDA_CLASS).nonzero().squeeze()
        if fda_indices.dim() == 0:
            fda_indices = fda_indices.unsqueeze(0)
        if fda_indices.numel() == 0:
            return torch.tensor(0.0, device=z.device)
        
        non_fda_indices = torch.tensor([i for i, l in enumerate(labels) if l in negative_labels], 
                                     device=z.device)
        if non_fda_indices.numel() == 0:
            return torch.tensor(0.0, device=z.device)
        
        triplet_loss = 0.0
        num_triplets = 0
        
        # Create triplets: (anchor, positive, negative)
        for anchor_idx in fda_indices:
            positive_candidates = [idx for idx in fda_indices if idx != anchor_idx]
            if len(positive_candidates) == 0:
                continue
            
            positive_idx = random.choice(positive_candidates)
            negative_idx = random.choice(non_fda_indices)
            
            anchor = z[anchor_idx]
            positive = z[positive_idx]
            negative = z[negative_idx]
            
            # Compute distances
            pos_dist = torch.sum((anchor - positive)**2)
            neg_dist = torch.sum((anchor - negative)**2)
            
            # Triplet loss with margin
            triplet_loss += torch.relu(pos_dist - neg_dist + self.margin)
            num_triplets += 1
        
        return triplet_loss / num_triplets if num_triplets > 0 else torch.tensor(0.0, device=z.device)
    
    def forward(self, recon_x, x, mu, log_var, z, labels):
        """Compute combined loss"""
        recon_loss = self.reconstruction_loss(recon_x, x)
        kl_loss = self.kl_divergence(mu, log_var)
        trip_loss = self.triplet_loss(z, labels)
        
        total_loss = (self.reconstruction_weight * recon_loss + 
                     self.kl_weight * kl_loss +
                     self.triplet_weight * trip_loss)
        
        return total_loss, recon_loss, kl_loss, trip_loss


def train_epoch(model, train_loader, criterion, optimizer, device='cpu'):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_triplet_loss = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon, mu, log_var, z = model(data)
        loss, recon_loss, kl_loss, trip_loss = criterion(
            recon, data, mu, log_var, z, labels
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_triplet_loss += trip_loss.item()
    
    num_batches = len(train_loader)
    return {
        'total': total_loss / num_batches,
        'reconstruction': total_recon_loss / num_batches,
        'kl': total_kl_loss / num_batches,
        'triplet': total_triplet_loss / num_batches
    }


def evaluate_clustering(model, data_loader, device='cpu'):
    """Evaluate clustering performance"""
    model.eval()
    all_embeddings = []
    all_labels = []
    reconstruction_error = 0
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            recon, mu, log_var, z = model(data)
            
            reconstruction_error += F.mse_loss(recon, data, reduction='sum').item()
            all_embeddings.append(z.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Hungarian algorithm for optimal cluster assignment
    conf_matrix = confusion_matrix(labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    
    accuracy = conf_matrix[row_ind, col_ind].sum() / labels.shape[0] * 100
    avg_reconstruction_error = reconstruction_error / len(data_loader.dataset)
    
    return embeddings, labels, accuracy, avg_reconstruction_error


def train_triplet_clvae(data_path, scaler_path, epochs=100, batch_size=64, latent_dim=3,
                      reconstruction_weight=0.5, triplet_weight=0.5, margin=1.0,
                      learning_rate=1e-3, device='cpu'):
    """
    Main training function for Triplet CVAE
    
    Args:
        data_path: Path to training data CSV
        scaler_path: Path to saved feature scalers
        epochs: Number of training epochs
        batch_size: Training batch size
        latent_dim: Dimensionality of latent space
        reconstruction_weight: Weight for reconstruction loss
        triplet_weight: Weight for triplet loss
        margin: Margin for triplet loss
        learning_rate: Learning rate for optimizer
        device: Training device ('cpu' or 'cuda')
    
    Returns:
        model: Trained model
        history: Training history
    """
    
    # Load data and scalers
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    
    data = pd.read_csv(data_path)
    X = data[FEATURES].copy()
    source = data['Source'].map(SOURCE_MAPPING)
    
    # Clean data
    clean_mask = ~X.isna().any(axis=1)
    X_clean = X[clean_mask]
    source_clean = source[clean_mask]
    
    # Apply scaling
    continuous_features = ['LogP', 'MolecularWeight', 'SA_Score', 'VDss']
    integer_features = ['HBD', 'HBA', 'ROTB']
    
    X_clean[continuous_features] = scalers['continuous_scaler'].transform(X_clean[continuous_features])
    X_clean[integer_features] = scalers['integer_scaler'].transform(X_clean[integer_features])
    
    # Create data loader
    X_tensor = torch.FloatTensor(X_clean.values)
    source_tensor = torch.LongTensor(source_clean.values)
    dataset = TensorDataset(X_tensor, source_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = TripletCLVAE(input_dim=len(FEATURES), latent_dim=latent_dim, 
                       hidden_dims=[128, 64])
    model.to(device)
    
    # Initialize loss and optimizer
    criterion = TripletVAELoss(reconstruction_weight=reconstruction_weight, 
                              kl_weight=1e-3, 
                              triplet_weight=triplet_weight,
                              margin=margin)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    history = []
    best_accuracy = 0
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(1, epochs + 1):
        # Train one epoch
        losses = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        history.append(losses)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] - "
                  f"Total Loss: {losses['total']:.4f}, "
                  f"Recon: {losses['reconstruction']:.4f}, "
                  f"KL: {losses['kl']:.4f}, "
                  f"Triplet: {losses['triplet']:.4f}")
            
            # Evaluate clustering
            embeddings, labels, accuracy, recon_error = evaluate_clustering(model, train_loader, device)
            print(f"Clustering Accuracy: {accuracy:.2f}%, "
                  f"Reconstruction Error: {recon_error:.6f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\nTraining completed!")
    print(f"Best clustering accuracy: {best_accuracy:.2f}%")
    
    return model, history


# Example usage
if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Train model
    model, history = train_triplet_clvae(
        data_path="all_data_train_set.csv",
        scaler_path="drug_scalersV2.pkl",
        epochs=100,
        batch_size=64,
        latent_dim=3,
        reconstruction_weight=0.5,
        triplet_weight=0.5,
        margin=1.0,
        learning_rate=1e-3,
        device=device
    )
    
    # Save trained model
    torch.save(model.state_dict(), 'triplet_clvae_model.pt')
    print("Model saved as 'triplet_clvae_model.pt'")