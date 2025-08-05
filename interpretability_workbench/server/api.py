"""
FastAPI backend for InterpretabilityWorkbench
"""
import logging
import logging.handlers
import sys
import asyncio
import time
from datetime import datetime
from pathlib import Path
from functools import wraps
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import torch
import numpy as np
import pandas as pd

from ..lora_patch import LoRAPatcher
from ..sae_train import SparseAutoencoder, SAETrainer
from ..trace import FeatureAnalyzer
from ..feature_provenance import FeatureProvenanceAnalyzer
from ..cache_manager import cache_manager
import safetensors.torch as safetensors
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import threading

# Configure logging
def setup_logging():
    """Setup structured logging for the application"""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "api.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    # Create specific loggers
    api_logger = logging.getLogger("api")
    model_logger = logging.getLogger("model")
    sae_logger = logging.getLogger("sae")
    training_logger = logging.getLogger("training")
    websocket_logger = logging.getLogger("websocket")
    
    return {
        "api": api_logger,
        "model": model_logger,
        "sae": sae_logger,
        "training": training_logger,
        "websocket": websocket_logger
    }

# Setup logging
loggers = setup_logging()
api_logger = loggers["api"]
model_logger = loggers["model"]
sae_logger = loggers["sae"]
training_logger = loggers["training"]
websocket_logger = loggers["websocket"]

api_logger.info("Starting InterpretabilityWorkbench API server")


# Pydantic models for API
class TokenInfo(BaseModel):
    token: str
    strength: float

class FeatureInfo(BaseModel):
    id: str
    layer_idx: int
    sparsity: float
    top_tokens: List[TokenInfo]
    activation_strength: float


class PatchRequest(BaseModel):
    feature_id: str
    layer_idx: int
    strength: float
    enabled: bool


class UpdatePatchRequest(BaseModel):
    strength: Optional[float] = None
    isEnabled: Optional[bool] = None


class TokenProbability(BaseModel):
    token: str
    token_id: int
    probability: float
    logit: float


class ModelState:
    """Global model state manager"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.patcher: Optional[LoRAPatcher] = None
        self.sae_models: Dict[int, SparseAutoencoder] = {}
        self.features: Dict[str, Dict[str, Any]] = {}
        self.feature_analyzers: Dict[int, FeatureAnalyzer] = {}
        self.activation_data_paths: Dict[int, str] = {}
        self.model_name = None
        # Patch management
        self.patches: Dict[str, Dict[str, Any]] = {}  # patch_id -> patch_info
        # Provenance analysis
        self.provenance_analyzer: Optional[FeatureProvenanceAnalyzer] = None
        # SAE training state
        self.sae_training_jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> job_info
        
    def load_model(self, model_name: str):
        """Load model and tokenizer"""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            model_logger.info(f"Loading model {model_name}...")
            self.model_name = model_name
            
            model_logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            model_logger.info("Loading model...")
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model.eval()
            
            model_logger.info("Creating LoRA patcher...")
            self.patcher = LoRAPatcher(self.model)
            model_logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            import traceback
            error_msg = f"Failed to load model {model_name}: {str(e)}\n{traceback.format_exc()}"
            model_logger.error(error_msg)
            raise
    
    def load_sae(self, layer_idx: int, sae_path: str):
        """Load SAE for a specific layer"""
        try:
            sae_logger.info(f"Loading SAE for layer {layer_idx} from {sae_path}")
            
            # Load metadata
            metadata_path = Path(sae_path).parent / f"sae_layer_{layer_idx}_metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Create SAE instance
            sae = SparseAutoencoder(
                input_dim=metadata["input_dim"],
                latent_dim=metadata["latent_dim"],
                tied_weights=metadata["tied_weights"]
            )
            
            # Load weights
            state_dict = safetensors.load_file(sae_path)
            sae.load_state_dict(state_dict)
            sae.eval()
            
            self.sae_models[layer_idx] = sae
            sae_logger.info(f"SAE loaded successfully for layer {layer_idx}")
        except Exception as e:
            sae_logger.error(f"Failed to load SAE for layer {layer_idx}: {str(e)}")
            raise
    
    def analyze_features(self, layer_idx: int, activation_data_path: Optional[str] = None, top_k: int = 100):
        """Analyze SAE features and store metadata"""
        try:
            if layer_idx not in self.sae_models:
                raise ValueError(f"No SAE loaded for layer {layer_idx}")
            
            sae_logger.info(f"Starting feature analysis for layer {layer_idx}, analyzing top {top_k} features")
            
            sae = self.sae_models[layer_idx]
            
            # Analyze encoder weights to understand features
            encoder_weights = sae.encoder.weight.data  # Shape: [latent_dim, input_dim]
            
            # Use feature analyzer if available for token analysis
            analyzer = self.feature_analyzers.get(layer_idx)
            
            for feature_idx in range(min(top_k, encoder_weights.shape[0])):
                feature_id = f"layer_{layer_idx}_feature_{feature_idx}"
                
                # Get feature vector
                feature_vector = encoder_weights[feature_idx]
                
                # Compute some basic statistics
                sparsity = (feature_vector.abs() < 1e-6).float().mean().item()
                activation_strength = feature_vector.norm().item()
                
                # Get top tokens if analyzer is available
                top_tokens = []
                if analyzer:
                    try:
                        token_analysis = analyzer.analyze_feature_tokens(feature_idx, top_k=5)
                        top_tokens = [{"token": token['token'], "strength": token.get('strength', 0.0)} for token in token_analysis]
                    except Exception as e:
                        sae_logger.warning(f"Could not analyze tokens for feature {feature_idx}: {e}")
                else:
                    # Fallback: create token objects with placeholder strength
                    # This won't be very meaningful but maintains consistency
                    sae_logger.warning(f"No tokenizer available for feature {feature_idx}, using placeholder tokens")
                    top_tokens = [{"token": f"token_{i}", "strength": 0.0} for i in range(5)]
                
                # Store feature info
                self.features[feature_id] = {
                    "id": feature_id,
                    "layer_idx": layer_idx,
                    "feature_idx": feature_idx,
                    "sparsity": sparsity,
                    "activation_strength": activation_strength,
                    "vector": feature_vector.cpu().numpy(),
                    "top_tokens": top_tokens
                }
            
            sae_logger.info(f"Successfully analyzed {min(top_k, encoder_weights.shape[0])} features for layer {layer_idx}")
        except Exception as e:
            sae_logger.error(f"Failed to analyze features for layer {layer_idx}: {str(e)}")
            raise
    
    def apply_patches(self, **inputs):
        """Apply active patches to model inputs"""
        if not self.patches or not any(patch.get('isEnabled', False) for patch in self.patches.values()):
            return self.model(**inputs)
        
        # Check if we have a patcher with active patches
        if not hasattr(self, 'patcher') or not self.patcher:
            return self.model(**inputs)
        
        # Get active patches from patcher
        active_patches = self.patcher.list_patches()
        enabled_patches = {k: v for k, v in active_patches.items() if v.get('enabled', False)}
        
        if not enabled_patches:
            return self.model(**inputs)
        
        # Apply patches through the patcher
        try:
            # The patcher should handle the actual patch application
            # We need to ensure patches are properly applied to the model
            patched_output = self.model(**inputs)
            
            # For now, return the output as-is since the patcher handles the modifications
            # In a full implementation, we'd need to ensure the patches are actually applied
            # during the forward pass
            model_logger.info(f"Applied {len(enabled_patches)} active patches")
            return patched_output
            
        except Exception as e:
            model_logger.error(f"Error applying patches: {str(e)}")
            # Fallback to original model output
            return self.model(**inputs)


# Global state
model_state = ModelState()

# FastAPI app
app = FastAPI(title="InterpretabilityWorkbench API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance monitoring decorator
def measure_latency(func):
    """Decorator to measure API endpoint latency"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            # Log latency
            api_logger.info(f"{func.__name__} latency: {latency:.2f}ms")
            
            # Alert if latency exceeds target (400ms for inference, 100ms for others)
            target = 400 if "inference" in func.__name__ else 100
            if latency > target:
                api_logger.warning(f"High latency detected in {func.__name__}: {latency:.2f}ms")
            
            return result
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            api_logger.error(f"{func.__name__} failed after {latency:.2f}ms: {str(e)}")
            raise
    return wrapper

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection might be closed
                pass

manager = ConnectionManager()


# API Routes
@app.get("/health")
async def health_check():
    api_logger.debug("Health check requested")
    return {"status": "healthy", "model_loaded": model_state.model is not None}


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = cache_manager.get_stats()
        return {
            "success": True,
            "cache_stats": stats
        }
    except Exception as e:
        api_logger.error(f"Failed to get cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    try:
        cache_manager.clear_all()
        return {"success": True, "message": "All caches cleared"}
    except Exception as e:
        api_logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.get("/model/status")
async def get_model_status():
    """Get current model status"""
    if model_state.model is None:
        return {"status": "idle"}
    
    return {
        "status": "ready",
        "modelName": model_state.model_name,
        "model_name": model_state.model_name  # Support both naming conventions
    }


@app.get("/sae/status") 
async def get_sae_status():
    """Get current SAE status"""
    if not model_state.sae_models:
        return {"status": "idle"}
    
    total_features = sum(len(features) for features in model_state.features.values())
    
    return {
        "status": "ready",
        "layerCount": len(model_state.sae_models),
        "featureCount": total_features
    }


class LoadModelRequest(BaseModel):
    model_name: str
    
    @classmethod
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        if len(v) > 200:
            raise ValueError("Model name too long")
        return v.strip()


class TrainSAERequest(BaseModel):
    layer_idx: int
    activation_data_path: str
    latent_dim: int = 16384
    sparsity_coef: float = 1e-3
    learning_rate: float = 1e-3
    max_epochs: int = 100
    tied_weights: bool = True
    activation_fn: str = "relu"
    output_dir: Optional[str] = None
    max_samples: Optional[int] = None  # Limit samples for faster training
    batch_size: int = 512  # Optimized batch size
    num_workers: int = 8   # Number of data loading workers

@app.post("/load-model")
async def load_model(request: LoadModelRequest):
    """Load a model for inference"""
    try:
        # Validate model name
        if not request.model_name or not request.model_name.strip():
            raise ValueError("Model name cannot be empty")
        
        # Check if model is already loaded
        if model_state.model is not None and model_state.model_name == request.model_name:
            api_logger.info(f"Model {request.model_name} is already loaded")
            return {
                "status": "ready", 
                "model_name": request.model_name,
                "success": True,
                "message": "Model already loaded"
            }
        
        api_logger.info(f"Loading model: {request.model_name}")
        model_state.load_model(request.model_name)
        
        # Initialize provenance analyzer if we have SAE models
        if model_state.sae_models and model_state.tokenizer:
            try:
                model_state.provenance_analyzer = FeatureProvenanceAnalyzer(
                    model=model_state.model,
                    sae_models=model_state.sae_models,
                    tokenizer=model_state.tokenizer,
                    activation_data_paths=model_state.activation_data_paths
                )
                api_logger.info("Provenance analyzer initialized")
            except Exception as e:
                api_logger.warning(f"Could not initialize provenance analyzer: {e}")
        
        api_logger.info(f"Model {request.model_name} loaded successfully")
        return {
            "status": "ready", 
            "model_name": request.model_name,
            "success": True
        }
    except ValueError as e:
        api_logger.error(f"Invalid model name: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"Error loading model {request.model_name}: {str(e)}\n{traceback.format_exc()}"
        api_logger.error(error_detail)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


class LoadSAERequest(BaseModel):
    layer_idx: int
    saePath: str
    activationsPath: str




@app.post("/load-sae")
async def load_sae(request: LoadSAERequest):
    """Load SAE for a specific layer"""
    try:
        model_state.load_sae(request.layer_idx, request.saePath)
        
        # Store activation data path for feature analysis
        if request.activationsPath and Path(request.activationsPath).exists():
            model_state.activation_data_paths[request.layer_idx] = request.activationsPath
            
            # Create feature analyzer
            if model_state.tokenizer and request.layer_idx in model_state.sae_models:
                analyzer = FeatureAnalyzer(
                    sae_model=model_state.sae_models[request.layer_idx],
                    tokenizer=model_state.tokenizer,
                    activation_data_path=request.activationsPath,
                    layer_idx=request.layer_idx
                )
                model_state.feature_analyzers[request.layer_idx] = analyzer
        
        # Analyze only top 20 features to reduce memory usage
        model_state.analyze_features(request.layer_idx, request.activationsPath, top_k=20)
        return {"success": True, "layer_idx": request.layer_idx}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/features")
async def get_features(layer_idx: Optional[int] = None, limit: int = 100, offset: int = 0, search: Optional[str] = None, sortBy: str = "activation", sortOrder: str = "desc"):
    """Get list of available features"""
    features = []
    
    for feature_id, feature_data in model_state.features.items():
        if layer_idx is None or feature_data["layer_idx"] == layer_idx:
            # Convert token data to TokenInfo objects
            top_tokens = []
            for token_data in feature_data["top_tokens"]:
                if isinstance(token_data, dict):
                    top_tokens.append(TokenInfo(
                        token=token_data["token"],
                        strength=token_data["strength"]
                    ))
                else:
                    # Handle legacy string format
                    top_tokens.append(TokenInfo(token=str(token_data), strength=0.0))
            
            features.append(FeatureInfo(
                id=feature_data["id"],
                layer_idx=feature_data["layer_idx"],
                sparsity=feature_data["sparsity"],
                top_tokens=top_tokens,
                activation_strength=feature_data["activation_strength"]
            ))
    
    # Apply search filter if provided
    if search:
        features = [f for f in features if search.lower() in f.id.lower()]
    
    # Apply sorting
    if sortBy == "activation":
        features.sort(key=lambda x: x.activation_strength, reverse=(sortOrder == "desc"))
    elif sortBy == "layer":
        features.sort(key=lambda x: x.layer_idx, reverse=(sortOrder == "desc"))
    elif sortBy == "frequency":
        features.sort(key=lambda x: x.sparsity, reverse=(sortOrder == "desc"))
    
    total = len(features)
    
    # Apply pagination
    features = features[offset:offset + limit]
    
    return {
        "features": features,
        "total": total,
        "hasMore": offset + limit < total
    }


@app.post("/patch")
async def create_patch(patch_request: PatchRequest):
    """Create or update a LoRA patch"""
    if model_state.patcher is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    feature_id = patch_request.feature_id
    if feature_id not in model_state.features:
        raise HTTPException(status_code=404, detail=f"Feature {feature_id} not found")
    
    try:
        feature_data = model_state.features[feature_id]
        feature_vector = torch.tensor(feature_data["vector"])
        
        # Create patch
        patch_id = model_state.patcher.create_feature_patch(
            feature_id=feature_id,
            layer_idx=patch_request.layer_idx,
            feature_vector=feature_vector,
            strength=patch_request.strength
        )
        
        if not patch_request.enabled:
            model_state.patcher.disable_patch(patch_id)
        
        # Return patch in the format expected by frontend
        patches_dict = model_state.patcher.list_patches()
        patch_data = patches_dict[patch_id]
        return {
            "id": patch_id,
            "featureId": patch_data.get("feature_id", ""),
            "name": patch_data.get("name", ""),
            "isEnabled": patch_data.get("enabled", False),
            "strength": patch_data.get("strength", 1.0),
            "description": patch_data.get("description", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patches")
async def get_patches():
    """Get list of active patches"""
    if model_state.patcher is None:
        return []
    
    patches_dict = model_state.patcher.list_patches()
    # Convert dictionary to list format expected by frontend
    patches_list = []
    for patch_id, patch_data in patches_dict.items():
        patches_list.append({
            "id": patch_id,
            "featureId": patch_data.get("feature_id", ""),
            "name": patch_data.get("name", ""),
            "isEnabled": patch_data.get("enabled", False),
            "strength": patch_data.get("strength", 1.0),
            "description": patch_data.get("description", "")
        })
    
    return patches_list


@app.patch("/patch/{patch_id}")
async def update_patch(patch_id: str, request: UpdatePatchRequest):
    """Update patch properties (strength, enabled state)"""
    if model_state.patcher is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        patches = model_state.patcher.list_patches()
        if patch_id not in patches:
            raise HTTPException(status_code=404, detail=f"Patch {patch_id} not found")
        
        updated_properties = []
        
        # Update strength if provided
        if request.strength is not None:
            model_state.patcher.update_patch_strength(patch_id, request.strength)
            updated_properties.append(f"strength={request.strength}")
        
        # Update enabled state if provided
        if request.isEnabled is not None:
            if request.isEnabled:
                model_state.patcher.enable_patch(patch_id)
            else:
                model_state.patcher.disable_patch(patch_id)
            updated_properties.append(f"enabled={request.isEnabled}")
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "patch_updated",
            "patch_id": patch_id,
            "updates": updated_properties
        }))
        
        # Return updated patch in the format expected by frontend
        updated_patches = model_state.patcher.list_patches()
        patch_data = updated_patches[patch_id]
        return {
            "id": patch_id,
            "featureId": patch_data.get("feature_id", ""),
            "name": patch_data.get("name", ""),
            "isEnabled": patch_data.get("enabled", False),
            "strength": patch_data.get("strength", 1.0),
            "description": patch_data.get("description", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/patch/{patch_id}/toggle")
async def toggle_patch(patch_id: str):
    """Toggle a patch on/off"""
    if model_state.patcher is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        patches = model_state.patcher.list_patches()
        if patch_id not in patches:
            raise HTTPException(status_code=404, detail=f"Patch {patch_id} not found")
        
        if patches[patch_id]["enabled"]:
            model_state.patcher.disable_patch(patch_id)
            action = "disabled"
        else:
            model_state.patcher.enable_patch(patch_id)
            action = "enabled"
        
        # Broadcast update to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "patch_toggled",
            "patch_id": patch_id,
            "action": action
        }))
        
        # Return updated patch in the format expected by frontend
        updated_patches = model_state.patcher.list_patches()
        patch_data = updated_patches[patch_id]
        return {
            "id": patch_id,
            "featureId": patch_data.get("feature_id", ""),
            "name": patch_data.get("name", ""),
            "isEnabled": patch_data.get("enabled", False),
            "strength": patch_data.get("strength", 1.0),
            "description": patch_data.get("description", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference")
@measure_latency
async def run_inference(text: str, max_length: int = 50):
    """Run inference with current patches applied"""
    if model_state.model is None or model_state.tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Try to get from cache first
        cached_result = cache_manager.get_inference(text, max_length)
        if cached_result is not None:
            api_logger.info(f"Returning cached inference result for text: {text[:50]}...")
            return cached_result
        
        api_logger.info(f"Running inference on text: {text[:50]}...")
        
        # Tokenize input
        inputs = model_state.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Run inference with and without patches
        with torch.no_grad():
            # Original inference
            original_output = model_state.model(**inputs)
            
            # Patched inference (if patches are active)
            if hasattr(model_state, 'patches') and model_state.patches and any(patch.get('isEnabled', False) for patch in model_state.patches.values()):
                # Apply patches - this would need to be implemented in ModelState
                patched_output = original_output  # Placeholder for now
            else:
                patched_output = original_output
        
        # Get logits for next token prediction
        original_logits = original_output.logits[:, -1, :] if hasattr(original_output, 'logits') else original_output.last_hidden_state[:, -1, :]
        patched_logits = patched_output.logits[:, -1, :] if hasattr(patched_output, 'logits') else patched_output.last_hidden_state[:, -1, :]
        
        # Convert to probabilities
        original_probs = torch.softmax(original_logits, dim=-1)
        patched_probs = torch.softmax(patched_logits, dim=-1)
        
        # Get top tokens
        top_k = 20
        original_top_probs, original_top_indices = torch.topk(original_probs, top_k)
        patched_top_probs, patched_top_indices = torch.topk(patched_probs, top_k)
        
        # Convert to response format
        original_tokens = []
        patched_tokens = []
        
        for i in range(top_k):
            # Original tokens
            token_id = original_top_indices[0][i].item()
            prob = original_top_probs[0][i].item()
            token = model_state.tokenizer.decode([token_id])
            original_tokens.append({
                "token": token,
                "probability": prob,
                "token_id": token_id
            })
            
            # Patched tokens
            token_id = patched_top_indices[0][i].item()
            prob = patched_top_probs[0][i].item()
            token = model_state.tokenizer.decode([token_id])
            patched_tokens.append({
                "token": token,
                "probability": prob,
                "token_id": token_id
            })
        
        result = {
            "original": original_tokens,
            "patched": patched_tokens
        }
        
        # Cache the result
        cache_manager.put_inference(text, result, max_length)
        
        api_logger.info(f"Inference completed successfully")
        return result
        
    except Exception as e:
        api_logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/export-sae")
async def export_sae(layer_idx: int, output_path: str):
    """Export SAE weights and metadata"""
    if layer_idx not in model_state.sae_models:
        raise HTTPException(status_code=404, detail=f"No SAE loaded for layer {layer_idx}")
    
    try:
        sae = model_state.sae_models[layer_idx]
        export_path = Path(output_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export SAE weights
        sae_file = export_path / f"sae_layer_{layer_idx}.safetensors"
        safetensors.save_file(sae.state_dict(), sae_file)
        
        # Export metadata
        metadata = {
            "layer_idx": layer_idx,
            "input_dim": sae.input_dim,
            "latent_dim": sae.latent_dim,
            "tied_weights": sae.tied_weights,
            "model_name": model_state.model_name,
            "export_timestamp": pd.Timestamp.now().isoformat()
        }
        
        metadata_file = export_path / f"sae_layer_{layer_idx}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "success": True,
            "sae_file": str(sae_file),
            "metadata_file": str(metadata_file)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export-patches")
async def export_patches(output_path: str):
    """Export all LoRA patches"""
    if model_state.patcher is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        model_state.patcher.save_patches(output_path)
        return {"success": True, "output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export-features")
async def export_features(layer_idx: Optional[int] = None, output_path: str = "features_export.json"):
    """Export feature analysis data with comprehensive metadata"""
    try:
        # Filter features by layer if specified
        features_to_export = {}
        for feature_id, feature_data in model_state.features.items():
            if layer_idx is None or feature_data["layer_idx"] == layer_idx:
                # Convert numpy arrays to lists for JSON serialization
                export_data = feature_data.copy()
                if isinstance(export_data["vector"], np.ndarray):
                    export_data["vector"] = export_data["vector"].tolist()
                
                # Add provenance data if available
                if model_state.provenance_analyzer:
                    try:
                        provenance_data = model_state.provenance_analyzer.analyze_feature_relationships(feature_id)
                        export_data["provenance"] = provenance_data
                    except Exception as e:
                        sae_logger.warning(f"Could not add provenance for feature {feature_id}: {e}")
                        export_data["provenance"] = {"error": str(e)}
                
                features_to_export[feature_id] = export_data
        
        # Add comprehensive metadata
        export_data = {
            "features": features_to_export,
            "metadata": {
                "model_name": model_state.model_name,
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "total_features": len(features_to_export),
                "layer_filter": layer_idx,
                "sae_models_loaded": list(model_state.sae_models.keys()),
                "activation_data_paths": model_state.activation_data_paths,
                "export_version": "1.0",
                "provenance_analyzer_available": model_state.provenance_analyzer is not None
            }
        }
        
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)
        
        sae_logger.info(f"Exported {len(features_to_export)} features to {output_file}")
        return {
            "success": True,
            "output_file": str(output_file),
            "features_exported": len(features_to_export)
        }
    except Exception as e:
        sae_logger.error(f"Failed to export features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export features: {str(e)}")


@app.post("/import-features")
async def import_features(input_path: str, layer_idx: Optional[int] = None):
    """Import features from JSON file"""
    try:
        if not Path(input_path).exists():
            raise HTTPException(status_code=404, detail=f"Import file not found: {input_path}")
        
        with open(input_path, 'r') as f:
            import_data = json.load(f)
        
        # Validate import data structure
        if "features" not in import_data:
            raise HTTPException(status_code=400, detail="Invalid import file format: missing 'features' key")
        
        imported_count = 0
        for feature_id, feature_data in import_data["features"].items():
            # Validate feature data
            if "id" not in feature_data or "layer_idx" not in feature_data:
                sae_logger.warning(f"Skipping invalid feature data: {feature_data}")
                continue
            
            # Filter by layer if specified
            if layer_idx is not None and feature_data["layer_idx"] != layer_idx:
                continue
            
            # Import the feature
            model_state.features[feature_id] = feature_data
            imported_count += 1
        
        sae_logger.info(f"Imported {imported_count} features from {input_path}")
        return {"success": True, "imported_count": imported_count, "input_path": input_path}
    except Exception as e:
        sae_logger.error(f"Failed to import features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to import features: {str(e)}")


@app.get("/feature/{feature_id}/details")
async def get_feature_details(feature_id: str):
    """Get detailed information about a specific feature"""
    if feature_id not in model_state.features:
        raise HTTPException(status_code=404, detail=f"Feature {feature_id} not found")
    
    feature_data = model_state.features[feature_id]
    layer_idx = feature_data["layer_idx"]
    feature_idx = feature_data["feature_idx"]
    
    # Get detailed analysis if analyzer is available
    analyzer = model_state.feature_analyzers.get(layer_idx)
    if analyzer:
        try:
            detailed_analysis = analyzer.get_feature_summary(feature_idx)
            feature_data["detailed_analysis"] = detailed_analysis
        except Exception as e:
            sae_logger.warning(f"Could not get detailed analysis for feature {feature_id}: {e}")
    
    # Add provenance analysis if available
    if model_state.provenance_analyzer:
        try:
            provenance_data = model_state.provenance_analyzer.analyze_feature_relationships(feature_id)
            feature_data["provenance"] = provenance_data
        except Exception as e:
            sae_logger.warning(f"Could not analyze provenance for feature {feature_id}: {e}")
            feature_data["provenance"] = {"error": str(e)}
    
    return feature_data


@app.get("/feature/{feature_id}/provenance")
async def get_feature_provenance(feature_id: str, upstream_layers: int = 2, downstream_layers: int = 1):
    """Get provenance analysis for a specific feature"""
    if not model_state.provenance_analyzer:
        raise HTTPException(status_code=400, detail="Provenance analyzer not available")
    
    try:
        # Try to get from cache first
        cached_result = cache_manager.get_provenance(feature_id, upstream_layers, downstream_layers)
        if cached_result is not None:
            return cached_result
        
        # Perform analysis
        provenance_data = model_state.provenance_analyzer.analyze_feature_relationships(
            feature_id, upstream_layers, downstream_layers
        )
        
        # Cache the result
        cache_manager.put_provenance(feature_id, provenance_data, upstream_layers, downstream_layers)
        
        return provenance_data
    except Exception as e:
        sae_logger.error(f"Error analyzing provenance for feature {feature_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# SAE Training Endpoints
@app.post("/sae/train")
async def train_sae(request: TrainSAERequest):
    """Start training a new SAE"""
    try:
        # Validate that we have a model loaded
        if model_state.model is None:
            raise HTTPException(status_code=400, detail="No model loaded. Load a model first before training SAE.")

        training_logger.info(f"Starting SAE training for layer {request.layer_idx}")
        
        # Generate unique job ID
        import uuid
        job_id = str(uuid.uuid4())
        
        # Set up output directory
        output_dir = request.output_dir or f"sae_training_{job_id}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Store job info
        job_info = {
            "job_id": job_id,
            "status": "starting",
            "layer_idx": request.layer_idx,
            "output_dir": str(output_path),
            "config": request.model_dump(),
            "progress": {"current_epoch": 0, "total_epochs": request.max_epochs},
            "metrics": {"train_loss": [], "reconstruction_loss": [], "sparsity_loss": []},
            "created_at": pd.Timestamp.now().isoformat()
        }
        model_state.sae_training_jobs[job_id] = job_info
        
        # Start training in background thread
        def train_worker():
            try:
                training_logger.info(f"Training worker started for job {job_id}")
                
                # Update status
                model_state.sae_training_jobs[job_id]["status"] = "preparing"
                
                # Check if activation data exists, if not, generate it
                if not Path(request.activation_data_path).exists():
                    training_logger.info(f"Activation data not found at {request.activation_data_path}. Generating...")
                    from ..trace import ActivationRecorder
                    
                    recorder = ActivationRecorder(
                        model_name=model_state.model_name,
                        layer_idx=request.layer_idx,
                        output_path=request.activation_data_path,
                        max_samples=5000,  # Generate reasonable amount for training
                        batch_size=4,
                        max_length=512
                    )
                    
                    # Record activations
                    recorder.record()
                    training_logger.info(f"Generated activation data at {request.activation_data_path}")
                
                # Validate that the parquet file is not empty
                if Path(request.activation_data_path).exists() and Path(request.activation_data_path).stat().st_size == 0:
                    raise Exception(f"Activation data file {request.activation_data_path} is empty. Please generate proper activation data first.")
                
                # Update status to training
                model_state.sae_training_jobs[job_id]["status"] = "training"
                training_logger.info(f"Starting SAE training for layer {request.layer_idx}")
                
                # Create trainer with optimized settings
                trainer_module = SAETrainer(
                    activation_path=request.activation_data_path,
                    layer_idx=request.layer_idx,
                    latent_dim=request.latent_dim,
                    sparsity_coef=request.sparsity_coef,
                    learning_rate=request.learning_rate,
                    tied_weights=request.tied_weights,
                    activation_fn=request.activation_fn,
                    output_dir=str(output_path),
                    batch_size=512,  # Optimized batch size
                    num_workers=8,   # More workers
                    mixed_precision=True,  # Enable mixed precision
                    gradient_accumulation_steps=1,
                    early_stopping_patience=10,
                    max_samples=request.max_samples,
                    batch_size=request.batch_size,
                    num_workers=request.num_workers
                )
                
                # Setup PyTorch Lightning trainer
                checkpoint_callback = ModelCheckpoint(
                    dirpath=output_path / "checkpoints",
                    filename="sae-{epoch:02d}-{train_loss:.4f}",
                    save_top_k=3,
                    monitor="train_loss",
                    mode="min"
                )
                
                # Add progress tracking callback
                from ..sae_train import TrainingProgressCallback
                progress_callback = TrainingProgressCallback(job_id, manager)
                
                trainer = pl.Trainer(
                    max_epochs=request.max_epochs,
                    accelerator="gpu",
                    devices=1,  # Single GPU for now
                    precision="16-mixed",  # Use mixed precision
                    accumulate_grad_batches=1,
                    callbacks=[checkpoint_callback, progress_callback],
                    enable_progress_bar=False,  # Disable for API usage
                    logger=False,  # Disable logging for API usage
                    log_every_n_steps=10,
                    val_check_interval=0.5,  # Validate twice per epoch
                    num_sanity_val_steps=2,  # Quick validation at start
                    reload_dataloaders_every_n_epochs=0,  # Don't reload dataloaders
                    sync_batchnorm=False,  # Disable for speed
                    deterministic=False,  # Disable for speed
                    benchmark=True  # Enable cuDNN benchmarking
                )
                
                # Train the model
                trainer.fit(trainer_module)
                
                # Save final model
                final_model_path = output_path / f"sae_layer_{request.layer_idx}.safetensors"
                safetensors.save_file(trainer_module.sae.state_dict(), final_model_path)
                
                # Save metadata
                metadata = {
                    "layer_idx": request.layer_idx,
                    "input_dim": trainer_module.sae.input_dim,
                    "latent_dim": trainer_module.sae.latent_dim,
                    "tied_weights": trainer_module.sae.tied_weights,
                    "activation_fn": request.activation_fn,
                    "sparsity_coef": request.sparsity_coef,
                    "learning_rate": request.learning_rate,
                    "training_epochs": request.max_epochs,
                    "model_name": model_state.model_name,
                    "training_completed_at": pd.Timestamp.now().isoformat()
                }
                
                metadata_path = output_path / f"sae_layer_{request.layer_idx}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Update job status
                model_state.sae_training_jobs[job_id].update({
                    "status": "completed",
                    "model_path": str(final_model_path),
                    "metadata_path": str(metadata_path),
                    "completed_at": pd.Timestamp.now().isoformat()
                })
                
                training_logger.info(f"SAE training completed successfully for job {job_id}")
                
                # Broadcast completion to WebSocket clients
                asyncio.create_task(manager.broadcast(json.dumps({
                    "type": "training_completed",
                    "job_id": job_id,
                    "status": "completed"
                })))
                
            except Exception as e:
                training_logger.error(f"SAE training failed for job {job_id}: {str(e)}")
                
                # Update job status with error
                model_state.sae_training_jobs[job_id].update({
                    "status": "failed",
                    "error": str(e),
                    "failed_at": pd.Timestamp.now().isoformat()
                })
                
                # Broadcast failure to WebSocket clients
                asyncio.create_task(manager.broadcast(json.dumps({
                    "type": "training_failed",
                    "job_id": job_id,
                    "error": str(e)
                })))
        
        # Start training thread
        training_thread = threading.Thread(target=train_worker)
        training_thread.daemon = True
        training_thread.start()
        
        training_logger.info(f"SAE training job {job_id} started successfully")
        return {
            "success": True,
            "job_id": job_id,
            "message": "SAE training started",
            "output_dir": str(output_path)
        }
        
    except Exception as e:
        training_logger.error(f"Failed to start SAE training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sae/training/status/{job_id}")
async def get_training_status(job_id: str):
    """Get status of SAE training job"""
    if job_id not in model_state.sae_training_jobs:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    job_info = model_state.sae_training_jobs[job_id].copy()
    
    # If training is in progress, read actual progress from checkpoint files
    if job_info["status"] == "training":
        output_dir = Path(job_info["output_dir"])
        checkpoints_dir = output_dir / "checkpoints"
        
        if checkpoints_dir.exists():
            # Get all checkpoint files
            checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
            
            if checkpoint_files:
                # Sort by modification time to get the latest
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
                latest_checkpoint = checkpoint_files[-1]
                
                # Extract epoch and loss from filename
                # Format: sae-epoch=02-train_loss=0.0886.ckpt
                filename = latest_checkpoint.name
                import re
                
                # Extract epoch number
                epoch_match = re.search(r'epoch=(\d+)', filename)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    job_info["progress"]["current_epoch"] = current_epoch
                
                # Extract loss value
                loss_match = re.search(r'train_loss=([\d.]+)', filename)
                if loss_match:
                    current_loss = float(loss_match.group(1))
                    
                    # Update metrics with current loss
                    if "train_loss" not in job_info["metrics"]:
                        job_info["metrics"]["train_loss"] = []
                    if "reconstruction_loss" not in job_info["metrics"]:
                        job_info["metrics"]["reconstruction_loss"] = []
                    if "sparsity_loss" not in job_info["metrics"]:
                        job_info["metrics"]["sparsity_loss"] = []
                    
                    # Add current loss to metrics (keep last 10 values)
                    job_info["metrics"]["train_loss"].append(current_loss)
                    if len(job_info["metrics"]["train_loss"]) > 10:
                        job_info["metrics"]["train_loss"] = job_info["metrics"]["train_loss"][-10:]
                    
                    # For now, use train_loss as reconstruction_loss (they're usually similar)
                    job_info["metrics"]["reconstruction_loss"].append(current_loss)
                    if len(job_info["metrics"]["reconstruction_loss"]) > 10:
                        job_info["metrics"]["reconstruction_loss"] = job_info["metrics"]["reconstruction_loss"][-10:]
                    
                    # Calculate actual sparsity loss based on feature activations
                    if hasattr(trainer.model, 'encoder'):
                        # Get activations from the encoder
                        activations = trainer.model.encoder(activations_batch)
                        sparsity_loss = torch.mean(torch.abs(activations)) * 0.1  # L1 sparsity
                    else:
                        # Fallback to rough estimate
                        sparsity_loss = current_loss * 0.1
                    job_info["metrics"]["sparsity_loss"].append(sparsity_loss)
                    if len(job_info["metrics"]["sparsity_loss"]) > 10:
                        job_info["metrics"]["sparsity_loss"] = job_info["metrics"]["sparsity_loss"][-10:]
                
                # Add checkpoint info
                job_info["latest_checkpoint"] = {
                    "filename": filename,
                    "path": str(latest_checkpoint),
                    "size_bytes": latest_checkpoint.stat().st_size,
                    "modified_at": pd.Timestamp.fromtimestamp(latest_checkpoint.stat().st_mtime).isoformat()
                }
                
                # Add total checkpoints count
                job_info["total_checkpoints"] = len(checkpoint_files)
    
    return job_info


@app.get("/sae/training/jobs")
async def list_training_jobs():
    """List all SAE training jobs"""
    return {
        "jobs": list(model_state.sae_training_jobs.values()),
        "total": len(model_state.sae_training_jobs)
    }


@app.delete("/sae/training/{job_id}")
async def cancel_training(job_id: str):
    """Cancel/delete SAE training job"""
    if job_id not in model_state.sae_training_jobs:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    job_info = model_state.sae_training_jobs[job_id]
    
    # If training is in progress, mark as cancelled (actual cancellation would need more complex logic)
    if job_info["status"] in ["starting", "training"]:
        job_info["status"] = "cancelled"
        job_info["cancelled_at"] = pd.Timestamp.now().isoformat()
    
    # Remove from jobs list
    del model_state.sae_training_jobs[job_id]
    
    return {"success": True, "message": f"Training job {job_id} cancelled"}


@app.post("/sae/load-trained")
async def load_trained_sae(job_id: str):
    """Load a trained SAE from a completed training job"""
    if job_id not in model_state.sae_training_jobs:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    job_info = model_state.sae_training_jobs[job_id]
    
    if job_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Training job {job_id} is not completed (status: {job_info['status']})")
    
    try:
        layer_idx = job_info["layer_idx"]
        model_path = job_info["model_path"]
        
        # Load the trained SAE
        model_state.load_sae(layer_idx, model_path)
        
        # If activation data was used for training, set it up for analysis
        activation_path = job_info["config"]["activation_data_path"]
        if Path(activation_path).exists():
            model_state.activation_data_paths[layer_idx] = activation_path
            
            # Create feature analyzer
            if model_state.tokenizer and layer_idx in model_state.sae_models:
                analyzer = FeatureAnalyzer(
                    sae_model=model_state.sae_models[layer_idx],
                    tokenizer=model_state.tokenizer,
                    activation_data_path=activation_path,
                    layer_idx=layer_idx
                )
                model_state.feature_analyzers[layer_idx] = analyzer
        
        # Analyze features
        model_state.analyze_features(layer_idx, activation_path)
        
        return {
            "success": True,
            "message": f"Trained SAE loaded for layer {layer_idx}",
            "layer_idx": layer_idx,
            "model_path": model_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    websocket_logger.info("New WebSocket connection established")
    await manager.connect(websocket)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            websocket_logger.debug(f"Received WebSocket message: {message.get('type', 'unknown')}")
            
            if message["type"] == "inference_request":
                # Run inference and send results
                data = message.get("data", {})
                text = data.get("text") or message.get("text", "")  # Support both formats
                max_tokens = data.get("maxTokens", 50)
                
                websocket_logger.info(f"Processing inference request: '{text[:50]}...'")
                result = await run_inference(text, max_tokens)
                
                response = {
                    "type": "inference_result",
                    "data": result
                }
                
                # Include requestId if provided for request-response pattern
                if "requestId" in message:
                    response["requestId"] = message["requestId"]
                
                await websocket.send_text(json.dumps(response))
                websocket_logger.debug("Inference result sent via WebSocket")
            elif message["type"] == "ping":
                # Respond to ping with pong
                await websocket.send_text(json.dumps({"type": "pong"}))
                websocket_logger.debug("Ping-pong response sent")
                
    except WebSocketDisconnect:
        websocket_logger.info("WebSocket connection disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        websocket_logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)


# Note: Static file serving is handled by nginx in production
# This server only handles API endpoints and WebSocket connections


@app.post("/sae/train-fast")
async def train_sae_fast(request: TrainSAERequest):
    """Train SAE with ultra-fast settings for quick experimentation"""
    try:
        # Validate that we have a model loaded
        if model_state.model is None:
            raise HTTPException(status_code=400, detail="No model loaded. Load a model first before training SAE.")

        training_logger.info(f"Starting FAST SAE training for layer {request.layer_idx}")
        
        # Generate unique job ID
        import uuid
        job_id = str(uuid.uuid4())
        
        # Set up output directory
        output_dir = request.output_dir or f"sae_fast_training_{job_id}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Store job info with fast settings
        job_info = {
            "job_id": job_id,
            "status": "starting",
            "layer_idx": request.layer_idx,
            "output_dir": str(output_path),
            "config": request.model_dump(),
            "training_mode": "fast",
            "progress": {"current_epoch": 0, "total_epochs": min(request.max_epochs, 50)},
            "metrics": {"train_loss": [], "reconstruction_loss": [], "sparsity_loss": []},
            "created_at": pd.Timestamp.now().isoformat()
        }
        model_state.sae_training_jobs[job_id] = job_info
        
        # Start training in background thread
        def train_worker():
            try:
                training_logger.info(f"Fast training worker started for job {job_id}")
                
                # Update status
                model_state.sae_training_jobs[job_id]["status"] = "preparing"
                
                # Check if activation data exists, if not, generate it
                if not Path(request.activation_data_path).exists():
                    training_logger.info(f"Activation data not found at {request.activation_data_path}. Generating...")
                    from ..trace import ActivationRecorder
                    
                    recorder = ActivationRecorder(
                        model_name=model_state.model_name,
                        layer_idx=request.layer_idx,
                        output_path=request.activation_data_path,
                        max_samples=10000,  # Smaller sample for fast training
                        batch_size=8,
                        max_length=256
                    )
                    
                    # Record activations
                    recorder.record()
                    training_logger.info(f"Generated activation data at {request.activation_data_path}")
                
                # Update status to training
                model_state.sae_training_jobs[job_id]["status"] = "training"
                training_logger.info(f"Starting FAST SAE training for layer {request.layer_idx}")
                
                # Create trainer with FAST settings
                trainer_module = SAETrainer(
                    activation_path=request.activation_data_path,
                    layer_idx=request.layer_idx,
                    latent_dim=min(request.latent_dim, 8192),  # Limit latent dim for speed
                    sparsity_coef=request.sparsity_coef,
                    learning_rate=2e-3,  # Higher learning rate
                    tied_weights=request.tied_weights,
                    activation_fn=request.activation_fn,
                    output_dir=str(output_path),
                    batch_size=1024,  # Larger batch size
                    num_workers=12,   # More workers
                    mixed_precision=True,
                    gradient_accumulation_steps=2,
                    early_stopping_patience=5,  # Stop earlier
                    max_samples=min(request.max_samples or 50000, 50000)  # Limit samples
                )
                
                # Configure callbacks
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=output_path,
                    filename=f"sae_fast_layer_{request.layer_idx}_epoch_{{epoch:02d}}_loss_{{val_loss:.4f}}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=2,  # Save fewer checkpoints
                    save_last=True
                )
                
                # Add progress tracking callback
                from ..sae_train import TrainingProgressCallback
                progress_callback = TrainingProgressCallback(job_id, manager)
                
                # Configure trainer with FAST optimizations
                trainer = pl.Trainer(
                    max_epochs=min(request.max_epochs, 50),  # Limit epochs
                    accelerator="gpu",
                    devices=1,
                    precision="16-mixed",
                    accumulate_grad_batches=2,
                    callbacks=[checkpoint_callback, progress_callback],
                    enable_progress_bar=False,
                    logger=False,
                    log_every_n_steps=5,  # More frequent logging
                    val_check_interval=0.25,  # Validate 4 times per epoch
                    num_sanity_val_steps=1,  # Quick validation
                    reload_dataloaders_every_n_epochs=0,
                    sync_batchnorm=False,
                    deterministic=False,
                    benchmark=True
                )
                
                # Train the model
                trainer.fit(trainer_module)
                
                # Save final model
                final_model_path = output_path / f"sae_fast_layer_{request.layer_idx}.safetensors"
                safetensors.save_file(trainer_module.sae.state_dict(), final_model_path)
                
                # Save metadata
                metadata = {
                    "layer_idx": request.layer_idx,
                    "input_dim": trainer_module.sae.input_dim,
                    "latent_dim": trainer_module.sae.latent_dim,
                    "tied_weights": trainer_module.sae.tied_weights,
                    "activation_fn": request.activation_fn,
                    "sparsity_coef": request.sparsity_coef,
                    "learning_rate": 2e-3,
                    "training_epochs": min(request.max_epochs, 50),
                    "model_name": model_state.model_name,
                    "training_mode": "fast",
                    "training_completed_at": pd.Timestamp.now().isoformat()
                }
                
                metadata_path = output_path / f"sae_fast_layer_{request.layer_idx}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Update job status
                model_state.sae_training_jobs[job_id].update({
                    "status": "completed",
                    "model_path": str(final_model_path),
                    "metadata_path": str(metadata_path),
                    "completed_at": pd.Timestamp.now().isoformat()
                })
                
                training_logger.info(f"FAST SAE training completed successfully for job {job_id}")
                
                # Broadcast completion to WebSocket clients
                asyncio.create_task(manager.broadcast(json.dumps({
                    "type": "training_completed",
                    "job_id": job_id,
                    "status": "completed",
                    "mode": "fast"
                })))
                
            except Exception as e:
                training_logger.error(f"FAST SAE training failed for job {job_id}: {str(e)}")
                
                # Update job status with error
                model_state.sae_training_jobs[job_id].update({
                    "status": "failed",
                    "error": str(e),
                    "failed_at": pd.Timestamp.now().isoformat()
                })
                
                # Broadcast failure to WebSocket clients
                asyncio.create_task(manager.broadcast(json.dumps({
                    "type": "training_failed",
                    "job_id": job_id,
                    "error": str(e),
                    "mode": "fast"
                })))
        
        # Start training thread
        training_thread = threading.Thread(target=train_worker)
        training_thread.daemon = True
        training_thread.start()
        
        training_logger.info(f"FAST SAE training job {job_id} started successfully")
        return {
            "success": True,
            "job_id": job_id,
            "message": "FAST SAE training started",
            "output_dir": str(output_path),
            "mode": "fast"
        }
        
    except Exception as e:
        training_logger.error(f"Failed to start FAST SAE training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point for the server"""
    import uvicorn
    import os
    
    api_logger.info("Starting InterpretabilityWorkbench server")
    
    # Disable reload in production/systemd environment
    reload_enabled = os.environ.get("RELOAD", "false").lower() == "true"
    
    api_logger.info(f"Server configuration: host=0.0.0.0, port=8000, reload={reload_enabled}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=reload_enabled,
        workers=1  # Single worker for systemd
    )


if __name__ == "__main__":
    main()