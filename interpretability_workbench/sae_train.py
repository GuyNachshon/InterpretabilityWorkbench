"""
Sparse Autoencoder Training with PyTorch Lightning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
from torch.utils.data import Dataset, DataLoader
import safetensors.torch as safetensors


class ActivationDataset(Dataset):
    """Dataset for loading activation data from parquet files"""
    
    def __init__(self, parquet_path: str, layer_idx: int):
        self.parquet_path = Path(parquet_path)
        self.layer_idx = layer_idx
        
        # Load the parquet file
        print(f"Loading activations from {parquet_path}...")
        table = pq.read_table(self.parquet_path)
        df = table.to_pandas()
        
        # Filter for the specific layer
        df = df[df['layer_idx'] == layer_idx]
        
        # Convert activation lists back to tensors
        self.activations = []
        for _, row in df.iterrows():
            act = torch.tensor(row['activation'], dtype=torch.float32)
            self.activations.append(act)
            
        self.activations = torch.stack(self.activations)
        print(f"Loaded {len(self.activations)} activations of shape {self.activations.shape}")
        
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx]


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with L1 sparsity penalty"""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        tied_weights: bool = True,
        activation_fn: str = "relu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.tied_weights = tied_weights
        
        # Encoder
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        
        # Decoder
        if tied_weights:
            self.decoder_weight = None  # Will use encoder.weight.T
        else:
            self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
            
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Activation function
        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation_fn}")
            
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        if not self.tied_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
            
    def encode(self, x):
        """Encode input to latent space"""
        return self.activation(self.encoder(x))
    
    def decode(self, z):
        """Decode latent representation back to input space"""
        if self.tied_weights:
            return F.linear(z, self.encoder.weight.t(), self.decoder_bias)
        else:
            return self.decoder(z) + self.decoder_bias
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class SAETrainer(pl.LightningModule):
    """PyTorch Lightning module for SAE training"""
    
    def __init__(
        self,
        activation_path: str,
        layer_idx: int,
        latent_dim: int = 16384,
        sparsity_coef: float = 1e-3,
        learning_rate: float = 1e-3,
        tied_weights: bool = True,
        activation_fn: str = "relu",
        output_dir: Optional[str] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.activation_path = activation_path
        self.layer_idx = layer_idx
        self.latent_dim = latent_dim
        self.sparsity_coef = sparsity_coef
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir) if output_dir else Path("sae_checkpoints")
        
        # Create dataset to get input dimension
        dataset = ActivationDataset(activation_path, layer_idx)
        input_dim = dataset.activations.shape[1]
        
        # Initialize SAE
        self.sae = SparseAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            tied_weights=tied_weights,
            activation_fn=activation_fn
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, x):
        return self.sae(x)
    
    def compute_loss(self, batch):
        """Compute reconstruction + sparsity loss"""
        x = batch
        x_recon, z = self.sae(x)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)
        
        # L1 sparsity penalty
        sparsity_loss = torch.mean(torch.abs(z))
        
        # Combined loss
        total_loss = recon_loss + self.sparsity_coef * sparsity_loss
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "sparsity_loss": sparsity_loss,
            "sparsity": torch.mean((z > 0).float())  # Fraction of active neurons
        }
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.compute_loss(batch)
        
        # Log metrics
        self.log("train_loss", loss_dict["loss"], prog_bar=True)
        self.log("train_recon", loss_dict["recon_loss"])
        self.log("train_sparsity_loss", loss_dict["sparsity_loss"])
        self.log("train_sparsity", loss_dict["sparsity"])
        
        return loss_dict["loss"]
    
    def validation_step(self, batch, batch_idx):
        loss_dict = self.compute_loss(batch)
        
        # Log metrics
        self.log("val_loss", loss_dict["loss"], prog_bar=True)
        self.log("val_recon", loss_dict["recon_loss"])
        self.log("val_sparsity_loss", loss_dict["sparsity_loss"])
        self.log("val_sparsity", loss_dict["sparsity"])
        
        return loss_dict["loss"]
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
    def setup(self, stage=None):
        """Setup datasets"""
        dataset = ActivationDataset(self.activation_path, self.layer_idx)
        
        # Split into train/val
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=256,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def on_train_end(self):
        """Save model weights in safetensors format"""
        self.output_dir.mkdir(exist_ok=True)
        
        # Save SAE weights
        sae_path = self.output_dir / f"sae_layer_{self.layer_idx}.safetensors"
        safetensors.save_file(
            self.sae.state_dict(),
            sae_path
        )
        
        # Save metadata
        metadata = {
            "layer_idx": self.layer_idx,
            "input_dim": self.sae.input_dim,
            "latent_dim": self.sae.latent_dim,
            "sparsity_coef": self.sparsity_coef,
            "tied_weights": self.sae.tied_weights,
            "final_train_loss": float(self.trainer.logged_metrics.get("train_loss", 0)),
            "final_val_loss": float(self.trainer.logged_metrics.get("val_loss", 0))
        }
        
        import json
        with open(self.output_dir / f"sae_layer_{self.layer_idx}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"SAE saved to {sae_path}")


def train_sae(
    activation_path: str,
    output_dir: str,
    layer_idx: int,
    latent_dim: int = 16384,
    sparsity_coef: float = 1e-3,
    max_epochs: int = 100,
    gpus: int = 1
):
    """Train a sparse autoencoder"""
    
    # Create trainer
    sae_trainer = SAETrainer(
        activation_path=activation_path,
        layer_idx=layer_idx,
        latent_dim=latent_dim,
        sparsity_coef=sparsity_coef,
        output_dir=output_dir
    )
    
    # Lightning trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=gpus,
        precision=16,  # Use mixed precision
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        check_val_every_n_epoch=1
    )
    
    # Train
    trainer.fit(sae_trainer)
    
    return sae_trainer


if __name__ == "__main__":
    # Test training
    trainer = train_sae(
        activation_path="test_activations.parquet",
        output_dir="test_sae_output",
        layer_idx=10,
        latent_dim=1024,  # Smaller for testing
        max_epochs=5
    )