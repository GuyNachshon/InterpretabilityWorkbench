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
import json
import asyncio
from datetime import datetime


class TrainingProgressCallback(pl.Callback):
    """Custom callback to track training progress and communicate via WebSocket"""
    
    def __init__(self, job_id: str, websocket_manager=None):
        super().__init__()
        self.job_id = job_id
        self.websocket_manager = websocket_manager
        self.start_time = datetime.now()
        
    def on_train_start(self, trainer, pl_module):
        """Called when training starts"""
        if self.websocket_manager:
            asyncio.create_task(self.websocket_manager.broadcast(json.dumps({
                "type": "training_progress",
                "job_id": self.job_id,
                "status": "training",
                "epoch": 0,
                "total_epochs": trainer.max_epochs,
                "progress": 0.0,
                "metrics": {}
            })))
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch"""
        current_epoch = trainer.current_epoch
        total_epochs = trainer.max_epochs
        progress = (current_epoch + 1) / total_epochs * 100
        
        # Get current metrics
        metrics = {}
        if hasattr(trainer, 'callback_metrics'):
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = value.item()
                else:
                    metrics[key] = value
        
        # Calculate estimated time remaining
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        if current_epoch > 0:
            avg_epoch_time = elapsed_time / (current_epoch + 1)
            remaining_epochs = total_epochs - (current_epoch + 1)
            estimated_remaining = avg_epoch_time * remaining_epochs
        else:
            estimated_remaining = 0
        
        if self.websocket_manager:
            asyncio.create_task(self.websocket_manager.broadcast(json.dumps({
                "type": "training_progress",
                "job_id": self.job_id,
                "status": "training",
                "epoch": current_epoch + 1,
                "total_epochs": total_epochs,
                "progress": progress,
                "metrics": metrics,
                "elapsed_time": elapsed_time,
                "estimated_remaining": estimated_remaining
            })))
    
    def on_train_end(self, trainer, pl_module):
        """Called when training ends"""
        if self.websocket_manager:
            asyncio.create_task(self.websocket_manager.broadcast(json.dumps({
                "type": "training_progress",
                "job_id": self.job_id,
                "status": "completed",
                "epoch": trainer.max_epochs,
                "total_epochs": trainer.max_epochs,
                "progress": 100.0,
                "metrics": {}
            })))


class ActivationDataset(Dataset):
    """Dataset for loading activation data from parquet files"""
    
    def __init__(self, parquet_path: str, layer_idx: int, max_samples: Optional[int] = None):
        self.parquet_path = Path(parquet_path)
        self.layer_idx = layer_idx
        
        # Load the parquet file
        print(f"Loading activations from {parquet_path}...")
        table = pq.read_table(self.parquet_path)
        df = table.to_pandas()
        
        # Filter for the specific layer
        df = df[df['layer_idx'] == layer_idx]
        
        # Limit samples if specified for faster training
        if max_samples and len(df) > max_samples:
            print(f"Limiting to {max_samples} samples for faster training")
            df = df.sample(n=max_samples, random_state=42)
        
        # Convert activation lists back to tensors more efficiently
        print(f"Converting {len(df)} activations to tensor...")
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
        output_dir: Optional[str] = None,
        batch_size: int = 512,  # Increased batch size for speed
        num_workers: int = 8,   # More workers for data loading
        mixed_precision: bool = True,  # Enable mixed precision
        gradient_accumulation_steps: int = 1,  # Gradient accumulation
        early_stopping_patience: int = 10,  # Early stopping
        lr_scheduler_patience: int = 5,  # LR scheduler patience
        warmup_epochs: int = 5,  # Learning rate warmup
        max_samples: Optional[int] = None  # Limit samples for faster training
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.activation_path = activation_path
        self.layer_idx = layer_idx
        self.latent_dim = latent_dim
        self.sparsity_coef = sparsity_coef
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir) if output_dir else Path("sae_checkpoints")
        
        # Performance optimization parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_patience = lr_scheduler_patience
        self.warmup_epochs = warmup_epochs
        self.max_samples = max_samples
        
        # Create dataset to get input dimension
        dataset = ActivationDataset(activation_path, layer_idx, max_samples=max_samples)
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
        # Use AdamW for better performance
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4,  # Add weight decay for regularization
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval each time
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
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
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch batches
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
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
    learning_rate: float = 1e-3,
    max_epochs: int = 100,
    gpus: int = 1,
    batch_size: int = 512,
    num_workers: int = 8,
    mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    early_stopping_patience: int = 10,
    max_samples: Optional[int] = None,
    strategy: str = "auto"
):
    """Train a sparse autoencoder with optimized settings for speed"""
    
    print(f"🚀 Starting optimized SAE training for layer {layer_idx}")
    print(f"📊 Configuration:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Workers: {num_workers}")
    print(f"   - Mixed precision: {mixed_precision}")
    print(f"   - Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   - Max samples: {max_samples or 'all'}")
    
    # Create trainer module with optimized settings
    trainer_module = SAETrainer(
        activation_path=activation_path,
        layer_idx=layer_idx,
        latent_dim=latent_dim,
        sparsity_coef=sparsity_coef,
        learning_rate=learning_rate,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        max_samples=max_samples
    )
    
    # Configure callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename=f"sae_layer_{layer_idx}_epoch_{{epoch:02d}}_loss_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min",
            verbose=True
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.ProgressBar(refresh_rate=10)  # Update progress bar more frequently
    ]
    
    # Configure trainer with optimizations
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else None,
        strategy=strategy,
        precision="16-mixed" if mixed_precision else "32",
        accumulate_grad_batches=gradient_accumulation_steps,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        num_sanity_val_steps=2,  # Quick validation at start
        reload_dataloaders_every_n_epochs=0,  # Don't reload dataloaders
        sync_batchnorm=False,  # Disable for speed
        deterministic=False,  # Disable for speed
        benchmark=True,  # Enable cuDNN benchmarking
        enable_checkpointing=True,
        enable_model_summary=True,
        max_time=None,  # No time limit
        limit_train_batches=None,  # Use all batches
        limit_val_batches=None,  # Use all validation batches
    )
    
    print(f"🎯 Training configuration:")
    print(f"   - Precision: {trainer.precision}")
    print(f"   - Strategy: {trainer.strategy}")
    print(f"   - Devices: {trainer.devices}")
    print(f"   - Accumulate grad batches: {trainer.accumulate_grad_batches}")
    
    # Train
    print(f"🚀 Starting training...")
    trainer.fit(trainer_module)
    
    print(f"✅ Training completed!")
    return trainer_module


def train_sae_fast(
    activation_path: str,
    output_dir: str,
    layer_idx: int,
    latent_dim: int = 8192,  # Smaller for speed
    max_samples: int = 50000,  # Limit samples
    max_epochs: int = 50,  # Fewer epochs
    gpus: int = 1
):
    """Ultra-fast SAE training with aggressive optimizations"""
    return train_sae(
        activation_path=activation_path,
        output_dir=output_dir,
        layer_idx=layer_idx,
        latent_dim=latent_dim,
        sparsity_coef=1e-3,
        learning_rate=2e-3,  # Higher learning rate
        max_epochs=max_epochs,
        gpus=gpus,
        batch_size=1024,  # Larger batch size
        num_workers=12,  # More workers
        mixed_precision=True,
        gradient_accumulation_steps=2,
        early_stopping_patience=5,  # Stop earlier
        max_samples=max_samples,
        strategy="ddp_find_unused_parameters_false" if gpus > 1 else "auto"
    )


if __name__ == "__main__":
    # Test training
    trainer = train_sae(
        activation_path="test_activations.parquet",
        output_dir="test_sae_output",
        layer_idx=10,
        latent_dim=1024,  # Smaller for testing
        max_epochs=5
    )