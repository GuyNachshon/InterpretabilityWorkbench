"""
InterpretabilityWorkbench: A mechanistic interpretability toolkit for LLMs
"""

__version__ = "0.1.0"

from .lora_patch import LoRAPatcher
from .sae_train import SparseAutoencoder
from .trace import FeatureAnalyzer, ActivationRecorder

__all__ = [
    "LoRAPatcher",
    "SparseAutoencoder", 
    "FeatureAnalyzer",
    "ActivationRecorder"
]