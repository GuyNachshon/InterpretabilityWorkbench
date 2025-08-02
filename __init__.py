"""
InterpretabilityWorkbench - Interactive Mechanistic-Interpretability Workbench for LLMs

A toolkit for recording activations, training sparse autoencoders, 
and performing live LoRA patching on language models.
"""

__version__ = "0.1.0"
__author__ = "InterpretabilityWorkbench Team"

from .trace import ActivationRecorder
from .sae_train import SparseAutoencoder, SAETrainer
from .lora_patch import LoRAPatcher, LoRAModule

__all__ = [
    "ActivationRecorder",
    "SparseAutoencoder", 
    "SAETrainer",
    "LoRAPatcher",
    "LoRAModule",
]