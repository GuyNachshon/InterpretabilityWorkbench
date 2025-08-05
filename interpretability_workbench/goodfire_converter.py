"""
Goodfire SAE Converter
Downloads Goodfire SAEs and converts them to the project's standard format
"""
import torch
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import safetensors.torch as safetensors


def download_and_convert_goodfire_sae(sae_name: str, output_dir: str = ".") -> str:
    """
    Download a Goodfire SAE and convert it to the project's standard format
    
    Args:
        sae_name: Name of the Goodfire SAE (e.g., 'Llama-3.1-8B-Instruct-SAE-l19')
        output_dir: Directory to save the converted files
    
    Returns:
        Path to the converted safetensors file
    """
    
    # SAE configurations
    SAE_CONFIGS = {
        'Llama-3.1-8B-Instruct-SAE-l19': {
            'repo_id': 'Goodfire/Llama-3.1-8B-Instruct-SAE-l19',
            'filename': 'Llama-3.1-8B-Instruct-SAE-l19.pth',
            'layer': 'model.layers.19',
            'expansion_factor': 16,
            'd_model': 4096,  # Llama 3.1 8B hidden size
        },
    }
    
    if sae_name not in SAE_CONFIGS:
        raise ValueError(f"Unknown SAE: {sae_name}. Available: {list(SAE_CONFIGS.keys())}")
    
    config = SAE_CONFIGS[sae_name]
    
    print(f"Downloading {sae_name} from {config['repo_id']}...")
    
    # Download the .pth file
    pth_path = hf_hub_download(
        repo_id=config['repo_id'],
        filename=config['filename'],
        repo_type="model"
    )
    
    print(f"Downloaded to: {pth_path}")
    
    # Load the PyTorch weights
    weights = torch.load(pth_path, weights_only=True, map_location='cpu')
    
    # Extract layer index from layer name
    layer_idx = int(config['layer'].split('.')[-1])
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as safetensors
    safetensors_path = output_path / f"sae_layer_{layer_idx}.safetensors"
    safetensors.save_file(weights, safetensors_path)
    
    # Create metadata file
    metadata = {
        "input_dim": config['d_model'],
        "latent_dim": config['d_model'] * config['expansion_factor'],
        "tied_weights": True,
        "activation_fn": "relu",
        "source": f"Goodfire/{sae_name}",
        "layer_idx": layer_idx,
        "expansion_factor": config['expansion_factor']
    }
    
    metadata_path = output_path / f"sae_layer_{layer_idx}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Converted to: {safetensors_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    return str(safetensors_path)


if __name__ == "__main__":
    # Example usage
    sae_path = download_and_convert_goodfire_sae("Llama-3.1-8B-Instruct-SAE-l19")
    print(f"Ready to use with: saePath={sae_path}") 