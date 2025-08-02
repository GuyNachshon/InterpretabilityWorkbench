"""
SAE Evaluation Script - Compute reconstruction metrics for trained sparse autoencoders
"""
import torch
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import safetensors.torch as safetensors
from torch.utils.data import Subset
from tqdm import tqdm

from sae_train import SparseAutoencoder, ActivationDataset


class SAEEvaluator:
    """Evaluates trained SAE models on held-out data"""
    
    def __init__(
        self,
        sae_path: str,
        activation_path: str,
        layer_idx: int,
        device: Optional[str] = None
    ):
        self.sae_path = Path(sae_path)
        self.activation_path = Path(activation_path)
        self.layer_idx = layer_idx
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Load SAE model
        self.sae = self._load_sae()
        self.sae.to(self.device)
        self.sae.eval()
        
        # Load evaluation dataset
        self.eval_dataset = self._load_eval_dataset()
        
    def _load_sae(self) -> SparseAutoencoder:
        """Load trained SAE model"""
        # Load metadata
        metadata_path = self.sae_path.parent / f"sae_layer_{self.layer_idx}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Create SAE instance
        sae = SparseAutoencoder(
            input_dim=metadata["input_dim"],
            latent_dim=metadata["latent_dim"],
            tied_weights=metadata.get("tied_weights", True),
            activation_fn=metadata.get("activation_fn", "relu")
        )
        
        # Load weights
        if not self.sae_path.exists():
            raise FileNotFoundError(f"SAE weights not found: {self.sae_path}")
        
        state_dict = safetensors.load_file(self.sae_path)
        sae.load_state_dict(state_dict)
        
        print(f"Loaded SAE from {self.sae_path}")
        print(f"Input dim: {metadata['input_dim']}, Latent dim: {metadata['latent_dim']}")
        
        return sae
    
    def _load_eval_dataset(self) -> Subset[Any]:
        """Load evaluation dataset (held-out portion)"""
        dataset = ActivationDataset(str(self.activation_path), self.layer_idx)
        
        # Use last 10% as eval set
        total_size = len(dataset)
        eval_start = int(0.9 * total_size)
        
        # Create subset
        eval_indices = list(range(eval_start, total_size))
        eval_subset = torch.utils.data.Subset(dataset, eval_indices)
        
        print(f"Evaluation dataset size: {len(eval_subset)} samples")
        return eval_subset
    
    def compute_reconstruction_metrics(self, batch_size: int = 256) -> Dict[str, float]:
        """Compute reconstruction loss and related metrics"""
        dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        total_recon_loss = 0.0
        total_l1_loss = 0.0
        total_samples = 0
        sparsity_values = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing metrics"):
                batch = batch.to(self.device)
                batch_size = batch.shape[0]
                
                # Forward pass
                x_recon, z = self.sae(batch)
                
                # Reconstruction loss (MSE)
                recon_loss = torch.nn.functional.mse_loss(x_recon, batch, reduction='sum')
                total_recon_loss += recon_loss.item()
                
                # L1 sparsity loss
                l1_loss = torch.sum(torch.abs(z))
                total_l1_loss += l1_loss.item()
                
                # Sparsity (fraction of active neurons)
                batch_sparsity = (z > 1e-6).float().mean(dim=0)  # Per feature
                sparsity_values.append(batch_sparsity.cpu())
                
                total_samples += batch_size
        
        # Aggregate metrics
        avg_recon_loss = total_recon_loss / total_samples
        avg_l1_loss = total_l1_loss / total_samples
        
        # Feature sparsity statistics
        all_sparsity = torch.cat(sparsity_values, dim=0)
        mean_sparsity = torch.mean(all_sparsity).item()
        
        # Compute explained variance
        explained_var = self._compute_explained_variance(dataloader)
        
        metrics = {
            "reconstruction_loss": avg_recon_loss,
            "l1_sparsity_loss": avg_l1_loss,
            "mean_feature_sparsity": mean_sparsity,
            "explained_variance": explained_var,
            "num_eval_samples": total_samples,
            "num_features": self.sae.latent_dim,
            "pass_threshold": avg_recon_loss <= 0.15  # From PLAN.md success criteria
        }
        
        return metrics
    
    def _compute_explained_variance(self, dataloader) -> float:
        """Compute explained variance R²"""
        original_var = 0.0
        residual_var = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                batch_size = batch.shape[0]
                
                # Forward pass
                x_recon, _ = self.sae(batch)
                
                # Variance of original data
                original_var += torch.var(batch, unbiased=False).item() * batch_size
                
                # Variance of residuals
                residuals = batch - x_recon
                residual_var += torch.var(residuals, unbiased=False).item() * batch_size
                
                total_samples += batch_size
        
        original_var /= total_samples
        residual_var /= total_samples
        
        # R² = 1 - (residual_var / original_var)
        explained_variance = 1.0 - (residual_var / original_var) if original_var > 0 else 0.0
        return explained_variance
    
    def analyze_feature_activations(self, top_k: int = 10) -> Dict[str, List[Dict]]:
        """Analyze which features activate most frequently"""
        dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=2
        )
        
        # Track feature activation counts and strengths
        feature_activations = torch.zeros(self.sae.latent_dim)
        feature_max_activations = torch.zeros(self.sae.latent_dim)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Analyzing features"):
                batch = batch.to(self.device)
                
                # Get feature activations
                _, z = self.sae(batch)
                
                # Count activations (features with activation > threshold)
                active_mask = z > 1e-6
                feature_activations += active_mask.sum(dim=0).cpu()
                
                # Track max activations
                batch_max = z.max(dim=0)[0].cpu()
                feature_max_activations = torch.maximum(feature_max_activations, batch_max)
        
        # Get top-k most active features
        _, top_indices = torch.topk(feature_activations, k=top_k)
        
        top_features = []
        for i, idx in enumerate(top_indices):
            feature_info = {
                "feature_idx": idx.item(),
                "activation_count": feature_activations[idx].item(),
                "max_activation": feature_max_activations[idx].item(),
                "activation_frequency": (feature_activations[idx] / len(self.eval_dataset)).item()
            }
            top_features.append(feature_info)
        
        # Get least active features (potential dead neurons)
        _, bottom_indices = torch.topk(feature_activations, k=top_k, largest=False)
        
        dead_features = []
        for idx in bottom_indices:
            if feature_activations[idx] == 0:
                dead_features.append({
                    "feature_idx": idx.item(),
                    "activation_count": 0,
                    "status": "dead"
                })
        
        return {
            "most_active_features": top_features,
            "dead_features": dead_features,
            "total_dead_features": (feature_activations == 0).sum().item()
        }
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict:
        """Generate comprehensive evaluation report"""
        print("Computing reconstruction metrics...")
        metrics = self.compute_reconstruction_metrics()
        
        print("Analyzing feature activations...")
        feature_analysis = self.analyze_feature_activations()
        
        # Combine into report
        report = {
            "model_info": {
                "sae_path": str(self.sae_path),
                "layer_idx": self.layer_idx,
                "input_dim": self.sae.input_dim,
                "latent_dim": self.sae.latent_dim,
                "tied_weights": self.sae.tied_weights
            },
            "reconstruction_metrics": metrics,
            "feature_analysis": feature_analysis,
            "evaluation_timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_path}")
        
        return report
    
    def print_summary(self, report: Dict):
        """Print summary of evaluation results"""
        metrics = report["reconstruction_metrics"]
        feature_analysis = report["feature_analysis"]
        
        print("\n" + "="*60)
        print("SAE EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Model: {report['model_info']['sae_path']}")
        print(f"Layer: {report['model_info']['layer_idx']}")
        print(f"Architecture: {report['model_info']['input_dim']} → {report['model_info']['latent_dim']}")
        
        print(f"\nReconstruction Metrics:")
        print(f"  MSE Loss: {metrics['reconstruction_loss']:.6f}")
        print(f"  Explained Variance (R²): {metrics['explained_variance']:.4f}")
        print(f"  Mean Feature Sparsity: {metrics['mean_feature_sparsity']:.4f}")
        print(f"  L1 Sparsity Loss: {metrics['l1_sparsity_loss']:.6f}")
        
        # Success criteria check
        threshold_met = "✅ PASS" if metrics['pass_threshold'] else "❌ FAIL"
        print(f"  Threshold (≤0.15): {threshold_met}")
        
        print(f"\nFeature Analysis:")
        print(f"  Total Features: {metrics['num_features']}")
        print(f"  Dead Features: {feature_analysis['total_dead_features']}")
        print(f"  Dead Feature Rate: {feature_analysis['total_dead_features']/metrics['num_features']:.2%}")
        
        print(f"\nTop 3 Most Active Features:")
        for i, feature in enumerate(feature_analysis['most_active_features'][:3]):
            print(f"  {i+1}. Feature {feature['feature_idx']}: "
                  f"{feature['activation_frequency']:.2%} activation rate, "
                  f"max={feature['max_activation']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SAE models")
    parser.add_argument("--sae-path", required=True, help="Path to SAE .safetensors file")
    parser.add_argument("--activations", required=True, help="Path to activations .parquet file")
    parser.add_argument("--layer", type=int, required=True, help="Layer index")
    parser.add_argument("--output", help="Output path for JSON report")
    parser.add_argument("--device", help="Device (cuda/mps/cpu)")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SAEEvaluator(
        sae_path=args.sae_path,
        activation_path=args.activations,
        layer_idx=args.layer,
        device=args.device
    )
    
    # Generate and print report
    report = evaluator.generate_report(args.output)
    evaluator.print_summary(report)


if __name__ == "__main__":
    main()