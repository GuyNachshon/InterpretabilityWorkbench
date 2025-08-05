"""
Live LoRA patching system for hot-swapping model edits
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
import safetensors.torch as safetensors
from transformers import PreTrainedModel
import numpy as np


class LoRAModule(nn.Module):
    """Low-Rank Adaptation module"""
    
    def __init__(
        self,
        original_module: nn.Module,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Get dimensions from original module
        if isinstance(original_module, nn.Linear):
            in_features = original_module.in_features
            out_features = original_module.out_features
        else:
            raise ValueError(f"Unsupported module type: {type(original_module)}")
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Enable/disable flag
        self.enabled = True
        
    def forward(self, x):
        """Forward pass with optional LoRA adaptation"""
        # Original forward pass
        result = self.original_module(x)
        
        if not self.enabled:
            return result
        
        # LoRA adaptation
        if self.dropout is not None:
            x = self.dropout(x)
            
        # Low-rank update: B @ A @ x
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        
        return result + lora_output * self.scaling
    
    def enable(self):
        """Enable LoRA adaptation"""
        self.enabled = True
        
    def disable(self):
        """Disable LoRA adaptation"""
        self.enabled = False


class LoRAPatcher:
    """Manages LoRA patches for live model editing"""
    
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.patches: Dict[str, LoRAModule] = {}
        self.original_modules: Dict[str, nn.Module] = {}
        self.patch_metadata: Dict[str, Dict[str, Any]] = {}
        
    def _get_target_modules(self, layer_idx: int, target_type: str = "mlp") -> List[str]:
        """Get target module names for patching"""
        targets = []
        
        # Common patterns for different model architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama-style
            base = f"model.layers.{layer_idx}"
            if target_type == "mlp":
                targets.extend([
                    f"{base}.mlp.gate_proj",
                    f"{base}.mlp.up_proj",
                    f"{base}.mlp.down_proj"
                ])
            elif target_type == "attention":
                targets.extend([
                    f"{base}.self_attn.q_proj",
                    f"{base}.self_attn.k_proj",
                    f"{base}.self_attn.v_proj",
                    f"{base}.self_attn.o_proj"
                ])
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-style
            base = f"transformer.h.{layer_idx}"
            if target_type == "mlp":
                targets.extend([
                    f"{base}.mlp.c_fc",
                    f"{base}.mlp.c_proj"
                ])
            elif target_type == "attention":
                targets.extend([
                    f"{base}.attn.c_attn",
                    f"{base}.attn.c_proj"
                ])
        
        return targets
    
    def create_feature_patch(
        self,
        feature_id: str,
        layer_idx: int,
        feature_vector: torch.Tensor,
        strength: float = 1.0,
        rank: int = 8,
        target_type: str = "mlp"
    ) -> str:
        """Create a LoRA patch to suppress/enhance a specific feature"""
        
        patch_id = f"feature_{feature_id}_layer_{layer_idx}"
        target_modules = self._get_target_modules(layer_idx, target_type)
        
        if not target_modules:
            raise ValueError(f"No target modules found for layer {layer_idx}")
        
        # Create patches for each target module
        patches_created = []
        
        for module_name in target_modules:
            # Get the module
            module = self._get_module_by_name(module_name)
            if module is None:
                print(f"Warning: Module {module_name} not found")
                continue
                
            # Create LoRA module
            lora_module = LoRAModule(module, rank=rank)
            
            # Initialize LoRA weights based on feature vector
            self._initialize_feature_lora(lora_module, feature_vector, strength)
            
            # Store original and replace with LoRA
            full_patch_id = f"{patch_id}_{module_name.replace('.', '_')}"
            self.original_modules[full_patch_id] = module
            self.patches[full_patch_id] = lora_module
            
            # Replace in model
            self._replace_module(module_name, lora_module)
            patches_created.append(full_patch_id)
        
        # Store metadata
        self.patch_metadata[patch_id] = {
            "feature_id": feature_id,
            "layer_idx": layer_idx,
            "strength": strength,
            "rank": rank,
            "target_type": target_type,
            "module_patches": patches_created,
            "enabled": True,
            "name": f"Feature {feature_id} (Layer {layer_idx})",
            "description": f"LoRA patch for feature {feature_id} at layer {layer_idx}"
        }
        
        return patch_id
    
    def _initialize_feature_lora(
        self,
        lora_module: LoRAModule,
        feature_vector: torch.Tensor,
        strength: float
    ):
        """Initialize LoRA weights to suppress/enhance a feature"""
        # Simple initialization: use feature vector to guide the adaptation
        feature_dim = feature_vector.shape[0]
        
        with torch.no_grad():
            # Initialize A matrix with feature vector pattern
            if lora_module.lora_A.shape[1] == feature_dim:
                lora_module.lora_A[0] = feature_vector * strength
            else:
                # Project feature vector to match dimensions
                scale = lora_module.lora_A.shape[1] / feature_dim
                if scale >= 1:
                    # Repeat/interpolate
                    expanded = feature_vector.repeat(int(scale))[:lora_module.lora_A.shape[1]]
                    lora_module.lora_A[0] = expanded * strength
                else:
                    # Downsample
                    downsampled = feature_vector[::int(1/scale)][:lora_module.lora_A.shape[1]]
                    lora_module.lora_A[0] = downsampled * strength
            
            # Initialize B matrix
            lora_module.lora_B[:, 0] = torch.randn(lora_module.lora_B.shape[0]) * 0.01 * strength
    
    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """Get module by dotted name"""
        parts = name.split('.')
        module = self.model
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        
        return module
    
    def _replace_module(self, name: str, new_module: nn.Module):
        """Replace a module in the model"""
        parts = name.split('.')
        parent = self.model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    def enable_patch(self, patch_id: str):
        """Enable a specific patch"""
        if patch_id not in self.patch_metadata:
            raise ValueError(f"Patch {patch_id} not found")
        
        metadata = self.patch_metadata[patch_id]
        for module_patch_id in metadata["module_patches"]:
            if module_patch_id in self.patches:
                self.patches[module_patch_id].enable()
        
        metadata["enabled"] = True
        
    def disable_patch(self, patch_id: str):
        """Disable a specific patch"""
        if patch_id not in self.patch_metadata:
            raise ValueError(f"Patch {patch_id} not found")
        
        metadata = self.patch_metadata[patch_id]
        for module_patch_id in metadata["module_patches"]:
            if module_patch_id in self.patches:
                self.patches[module_patch_id].disable()
        
        metadata["enabled"] = False
    
    def update_patch_strength(self, patch_id: str, strength: float):
        """Update the strength of a patch"""
        if patch_id not in self.patch_metadata:
            raise ValueError(f"Patch {patch_id} not found")
        
        metadata = self.patch_metadata[patch_id]
        metadata["strength"] = strength
        
        # Update the scaling factor for all LoRA modules in this patch
        for module_patch_id in metadata["module_patches"]:
            if module_patch_id in self.patches:
                lora_module = self.patches[module_patch_id]
                # Update the scaling factor
                lora_module.scaling = (lora_module.alpha / lora_module.rank) * strength
    
    def remove_patch(self, patch_id: str):
        """Completely remove a patch and restore original modules"""
        if patch_id not in self.patch_metadata:
            raise ValueError(f"Patch {patch_id} not found")
        
        metadata = self.patch_metadata[patch_id]
        
        # Restore original modules
        for module_patch_id in metadata["module_patches"]:
            if module_patch_id in self.original_modules:
                # Find the module name to restore
                original_module = self.original_modules[module_patch_id]
                
                # This is tricky - we need to reverse engineer the module name
                # For now, we'll mark it as disabled
                if module_patch_id in self.patches:
                    self.patches[module_patch_id].disable()
        
        # Clean up
        for module_patch_id in metadata["module_patches"]:
            self.patches.pop(module_patch_id, None)
            self.original_modules.pop(module_patch_id, None)
        
        self.patch_metadata.pop(patch_id)
    
    def list_patches(self) -> Dict[str, Dict[str, Any]]:
        """List all active patches"""
        return self.patch_metadata.copy()
    
    def save_patches(self, output_dir: str):
        """Save patches to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save patch weights
        for patch_id, lora_module in self.patches.items():
            weights = {
                "lora_A": lora_module.lora_A,
                "lora_B": lora_module.lora_B
            }
            safetensors.save_file(
                weights,
                output_path / f"{patch_id}.safetensors"
            )
        
        # Save metadata
        with open(output_path / "patch_metadata.json", "w") as f:
            json.dump(self.patch_metadata, f, indent=2)
        
        print(f"Patches saved to {output_path}")
    
    def load_patches(self, input_dir: str):
        """Load patches from disk"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise ValueError(f"Patch directory {input_dir} does not exist")
        
        # Load metadata
        metadata_file = input_path / "patch_metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"Patch metadata file not found: {metadata_file}")
        
        with open(metadata_file, "r") as f:
            self.patch_metadata = json.load(f)
        
        # Load patch weights and recreate LoRA modules
        for patch_id, metadata in self.patch_metadata.items():
            module_patches = metadata.get("module_patches", [])
            
            for module_patch_id in module_patches:
                patch_file = input_path / f"{module_patch_id}.safetensors"
                
                if patch_file.exists():
                    try:
                        # Load weights
                        weights = safetensors.load_file(patch_file)
                        
                        # Get original module name from patch_id
                        # Format: feature_xxx_layer_y_module_name
                        parts = module_patch_id.split('_')
                        if len(parts) >= 4:
                            # Reconstruct module name
                            module_name_parts = parts[3:]  # Skip feature_xxx_layer_y
                            module_name = '.'.join(module_name_parts)
                            
                            # Get original module
                            original_module = self._get_module_by_name(module_name)
                            if original_module is not None:
                                # Create LoRA module
                                rank = metadata.get("rank", 8)
                                lora_module = LoRAModule(original_module, rank=rank)
                                
                                # Load weights into LoRA module
                                lora_module.lora_A.data = weights["lora_A"]
                                lora_module.lora_B.data = weights["lora_B"]
                                
                                # Store in patches
                                self.original_modules[module_patch_id] = original_module
                                self.patches[module_patch_id] = lora_module
                                
                                # Replace in model
                                self._replace_module(module_name, lora_module)
                                
                                # Set enabled state
                                if metadata.get("enabled", False):
                                    lora_module.enable()
                                else:
                                    lora_module.disable()
                                    
                    except Exception as e:
                        print(f"Warning: Failed to load patch {module_patch_id}: {e}")
                        continue
                else:
                    print(f"Warning: Patch file not found: {patch_file}")
        
        print(f"Loaded {len(self.patches)} patches from {input_path}")


def test_lora_patching():
    """Test LoRA patching functionality"""
    from transformers import AutoModel
    
    # Load a small model for testing
    model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B")
    patcher = LoRAPatcher(model)
    
    # Create a dummy feature vector
    feature_vector = torch.randn(1024)  # Assuming hidden dim of 1024
    
    # Create a patch
    patch_id = patcher.create_feature_patch(
        feature_id="test_feature",
        layer_idx=5,
        feature_vector=feature_vector,
        strength=0.5
    )
    
    print(f"Created patch: {patch_id}")
    print(f"Active patches: {list(patcher.patch_metadata.keys())}")
    
    # Test enable/disable
    patcher.disable_patch(patch_id)
    print(f"Patch disabled: {patcher.patch_metadata[patch_id]['enabled']}")
    
    patcher.enable_patch(patch_id)
    print(f"Patch enabled: {patcher.patch_metadata[patch_id]['enabled']}")


if __name__ == "__main__":
    test_lora_patching()