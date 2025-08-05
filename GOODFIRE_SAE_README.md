# Goodfire SAE Integration

This project supports using pre-trained Sparse Autoencoders (SAEs) from Goodfire by converting them to the project's standard format. This allows you to use state-of-the-art interpretability models without training your own SAEs.

## Available SAEs

Currently supported SAEs:

- **Llama-3.1-8B-Instruct-SAE-l19**: Pre-trained SAE for layer 19 of Llama 3.1 8B Instruct model
  - Expansion factor: 16x
  - Layer: `model.layers.19`
  - Source: [Goodfire/Llama-3.1-8B-Instruct-SAE-l19](https://huggingface.co/Goodfire/Llama-3.1-8B-Instruct-SAE-l19)

## Quick Start

### 1. Install Dependencies

Make sure you have the required dependencies:

```bash
pip install huggingface-hub
```

### 2. Convert Goodfire SAE

First, convert the Goodfire SAE to the project's format:

```bash
python -m interpretability_workbench.goodfire_converter
```

This will download and convert the SAE to `sae_layer_19.safetensors` and `sae_layer_19_metadata.json`.

### 3. Load Model and SAE

#### Using the API

```python
import requests

# Load the model
response = requests.post("http://localhost:8000/load-model", json={
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct"
})

# Load the converted SAE (uses existing SAE loading)
response = requests.post("http://localhost:8000/load-sae", json={
    "layer_idx": 19,
    "saePath": "sae_layer_19.safetensors",
    "activationsPath": None  # Optional
})
```

#### Using the Test Script

```bash
python test_goodfire_sae.py
```

### 3. Using the Web UI

1. Load the model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
2. Use the new "Load Goodfire SAE" option with: `Llama-3.1-8B-Instruct-SAE-l19`

## Usage

### Convert Goodfire SAE

```bash
python -m interpretability_workbench.goodfire_converter
```

This creates:
- `sae_layer_19.safetensors` - The SAE weights in safetensors format
- `sae_layer_19_metadata.json` - Metadata file with SAE configuration

### Load SAE

Use the existing `/load-sae` endpoint with the converted file:

```json
{
  "layer_idx": 19,
  "saePath": "sae_layer_19.safetensors",
  "activationsPath": "optional/path/to/activations.parquet"
}
```

Response:
```json
{
  "success": true,
  "layer_idx": 19
}
```

## Configuration

The SAE configurations are defined in `interpretability_workbench/goodfire_sae.py`:

```python
SAE_CONFIGS = {
    'Llama-3.1-8B-Instruct-SAE-l19': {
        'repo_id': 'Goodfire/Llama-3.1-8B-Instruct-SAE-l19',
        'filename': 'Llama-3.1-8B-Instruct-SAE-l19.pth',
        'layer': 'model.layers.19',
        'expansion_factor': 16,
        'd_model': 4096,  # Llama 3.1 8B hidden size
    },
}
```

## Adding New SAEs

To add support for new Goodfire SAEs:

1. Add the configuration to `SAE_CONFIGS` in `goodfire_sae.py`
2. Update the documentation
3. Test with the provided test script

## Features

- **Automatic Download**: SAEs are automatically downloaded from Hugging Face and cached
- **Device Management**: Automatically uses the same device and dtype as the loaded model
- **Feature Analysis**: Integrates with the existing feature analysis system
- **Web UI Support**: Full integration with the web interface

## Troubleshooting

### Common Issues

1. **Model Not Loaded**: Make sure to load the model before loading the SAE
2. **Download Issues**: Check your internet connection and Hugging Face access
3. **Memory Issues**: Large models and SAEs require significant GPU memory

### Debug Mode

Enable debug logging by checking the service logs:

```bash
sudo journalctl -u interpretability-workbench -f
```

## References

- [Goodfire SAE Repository](https://huggingface.co/Goodfire/Llama-3.1-8B-Instruct-SAE-l19)
- [Goodfire Blog Post](https://www.goodfire.ai/blog/sae-open-source-announcement/)
- [Original Colab Notebook](https://colab.research.google.com/drive/1IBMQtJqy8JiRk1Q48jDEgTISmtxhlCRL#scrollTo=uGMosqFHdEXQ) 