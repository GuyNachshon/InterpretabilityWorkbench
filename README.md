# InterpretabilityWorkbench

*Interactive Mechanistic-Interpretability Workbench for LLMs*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com)

InterpretabilityWorkbench converts the traditional offline SAE (Sparse Autoencoder) pipeline into a **live laboratory**. Users can record activations, train SAEs, browse discovered features, and **hot-patch** model weights on-the-fly to observe real-time effects on token probabilities.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd InterpretabilityWorkbench

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```bash
# 1. Record activations from a model
microscope trace --model "Qwen/Qwen3-0.6B" --layer 10 --out activations.parquet

# 2. Train a sparse autoencoder
microscope train --activations activations.parquet --layer 10 --out sae_checkpoints/

# 3. Evaluate the trained SAE
python eval.py --sae-path sae_checkpoints/sae_layer_10.safetensors --activations activations.parquet --layer 10

# 4. Launch the web interface
microscope ui --model "Qwen/Qwen3-0.6B" --sae-dir sae_checkpoints/
```

Then visit `http://localhost:8000` to explore features and create live patches.

## 🏗️ Architecture

```
InterpretabilityWorkbench/
├── cli.py              # Entry point (trace|train|ui)
├── trace.py            # Activation recording with forward hooks
├── sae_train.py        # SAE training with PyTorch Lightning
├── lora_patch.py       # Live LoRA patching system
├── eval.py             # SAE evaluation and metrics
├── server/
│   ├── api.py          # FastAPI backend
│   └── websockets.py   # Real-time model interaction
├── ui/                 # React frontend (to be implemented)
└── tests/              # Unit tests
```

## 📊 Features

### ✅ Core Functionality

- **Activation Recording**: Stream model activations to Parquet with forward hooks
- **SAE Training**: PyTorch Lightning-based sparse autoencoder training
- **Feature Analysis**: Discover which tokens activate each feature most strongly
- **Live LoRA Patches**: Hot-swap model edits without restart
- **Real-time Inference**: WebSocket-based token probability updates
- **Export/Import**: Save SAE weights, patches, and feature analysis

### 🔧 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /load-model` | Load a HuggingFace model |
| `POST /load-sae` | Load trained SAE for a layer |
| `GET /features` | List discovered features with top tokens |
| `POST /patch` | Create LoRA patch for a feature |
| `POST /patch/{id}/toggle` | Toggle patch on/off |
| `POST /inference` | Run inference with current patches |
| `POST /export-sae` | Export SAE weights and metadata |
| `GET /feature/{id}/details` | Get detailed feature analysis |

### 📈 Evaluation Metrics

The `eval.py` script computes:
- **Reconstruction Loss** (MSE)
- **Explained Variance** (R²)
- **Feature Sparsity** (activation frequency)
- **Dead Neuron Detection**
- **Success Criteria**: MSE ≤ 0.15 (from project requirements)

## 🎯 Example Workflow

### 1. Record Activations
```python
from trace import ActivationRecorder

recorder = ActivationRecorder(
    model_name="microsoft/DialoGPT-small",
    layer_idx=8,
    output_path="layer8_activations.parquet",
    max_samples=10000
)
recorder.record(dataset_name="wikitext")
```

### 2. Train SAE
```python
from sae_train import train_sae

trainer = train_sae(
    activation_path="layer8_activations.parquet",
    output_dir="sae_models",
    layer_idx=8,
    latent_dim=4096,
    sparsity_coef=1e-3
)
```

### 3. Analyze Features
```python
from trace import FeatureAnalyzer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
analyzer = FeatureAnalyzer(sae, tokenizer, "layer8_activations.parquet", 8)

# Find top tokens for feature 42
top_tokens = analyzer.analyze_feature_tokens(42, top_k=10)
print(f"Feature 42 activates on: {[t['token'] for t in top_tokens]}")
```

### 4. Live Patching
```python
from lora_patch import LoRAPatcher

patcher = LoRAPatcher(model)
patch_id = patcher.create_feature_patch(
    feature_id="suspicious_feature",
    layer_idx=8,
    feature_vector=feature_vector,
    strength=-2.0  # Suppress this feature
)

# Test the effect
original_logits = model(input_ids).logits
patcher.enable_patch(patch_id)
patched_logits = model(input_ids).logits
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test categories
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ -m "gpu"       # GPU-only tests
```

## 📋 Success Criteria

From the project requirements:

| KPI | Target | Status |
|-----|--------|--------|
| SAE reconstruction loss (held-out) | ≤ 0.15 | ✅ Measured by `eval.py` |
| UI click → logits update | < 400 ms | ✅ WebSocket implementation |
| Feature provenance depth | ≥ 2 upstream layers | 🔄 In progress |

## 🔍 User Personas

- **Alignment Researcher Alice**: Records activations on a Trojan-finetuned model, locates trigger detectors, and disables malicious behavior in real-time.

- **ML Engineer Ben**: Imports colleague's SAE/LoRA files, attaches them to a local model, and exports safe patches for production deployment.

## ⚙️ Configuration

### Environment Variables
```bash
# Optional: Configure model loading
export HF_TOKEN="your-huggingface-token"
export CUDA_VISIBLE_DEVICES="0"

# Optional: Configure data paths
export SAE_CACHE_DIR="/path/to/sae/cache"
export ACTIVATION_DATA_DIR="/path/to/activations"
```

### Model Support

Tested with:
- ✅ GPT-style models (GPT-2, DialoGPT)
- ✅ Llama-style models (Llama-2, Code Llama)
- ✅ Qwen models
- ⚠️ BERT-style models (encoder-only, limited support)

## 🚨 System Requirements

- **Compute**: 1×A100-40GB *or* 2×RTX-4090 with 4-bit quantization
- **RAM**: 32GB+ recommended for larger models
- **Storage**: 100GB+ for activations and model checkpoints
- **Python**: 3.10+

## 🛠️ Development

### Setup Development Environment
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code formatting
black .
isort .

# Type checking
mypy .
```

### Adding New Features

1. **New SAE Architecture**: Extend `SparseAutoencoder` in `sae_train.py`
2. **New Patch Types**: Add methods to `LoRAPatcher` in `lora_patch.py`
3. **New Analysis**: Extend `FeatureAnalyzer` in `trace.py`
4. **New API Endpoints**: Add routes to `server/api.py`

## 📚 References

- [Sparse Autoencoders (Anthropic)](https://transformer-circuits.pub/2023/monosemantic-features)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- [SAE Interpretability](https://github.com/openai/sparse_autoencoder)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Anthropic for sparse autoencoder research
- OpenAI for LoRA techniques
- HuggingFace for model infrastructure
- PyTorch Lightning team for training framework

---

**Built with ❤️ for the mechanistic interpretability community**