"""
FastAPI backend for InterpretabilityWorkbench
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from ..lora_patch import LoRAPatcher
from ..sae_train import SparseAutoencoder, SAETrainer
from ..trace import FeatureAnalyzer
import safetensors.torch as safetensors
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import threading


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
        # SAE training state
        self.sae_training_jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> job_info
        
    def load_model(self, model_name: str):
        """Load model and tokenizer"""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            print(f"Loading model {model_name}...")
            self.model_name = model_name
            
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print("Loading model...")
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model.eval()
            
            print("Creating LoRA patcher...")
            self.patcher = LoRAPatcher(self.model)
            print(f"Model {model_name} loaded successfully")
        except Exception as e:
            import traceback
            error_msg = f"Failed to load model {model_name}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise
    
    def load_sae(self, layer_idx: int, sae_path: str):
        """Load SAE for a specific layer"""
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
        print(f"SAE loaded for layer {layer_idx}")
    
    def analyze_features(self, layer_idx: int, activation_data_path: Optional[str] = None, top_k: int = 100):
        """Analyze SAE features and store metadata"""
        if layer_idx not in self.sae_models:
            raise ValueError(f"No SAE loaded for layer {layer_idx}")
        
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
                    print(f"Warning: Could not analyze tokens for feature {feature_idx}: {e}")
            else:
                # Fallback: create token objects with placeholder strength
                # This won't be very meaningful but maintains consistency
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
        
        print(f"Analyzed {min(top_k, encoder_weights.shape[0])} features for layer {layer_idx}")


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
    return {"status": "healthy", "model_loaded": model_state.model is not None}


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

@app.post("/load-model")
async def load_model(request: LoadModelRequest):
    """Load a model for inference"""
    try:
        model_state.load_model(request.model_name)
        return {
            "status": "ready", 
            "model_name": request.model_name,
            "success": True
        }
    except Exception as e:
        import traceback
        error_detail = f"Error loading model {request.model_name}: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log to console
        raise HTTPException(status_code=500, detail=str(e))


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
async def run_inference(text: str, max_length: int = 50):
    """Run inference and return token probabilities"""
    if model_state.model is None or model_state.tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        # Tokenize input
        inputs = model_state.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Run inference
        with torch.no_grad():
            outputs = model_state.model(**inputs)
            
            # Get logits for next token prediction
            logits = outputs.last_hidden_state[:, -1, :]  # Last token logits
            
            # Get top-k probabilities
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=20)
            
            # Convert to response format
            token_probs = []
            for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
                token = model_state.tokenizer.decode([idx.item()])
                token_probs.append(TokenProbability(
                    token=token,
                    token_id=idx.item(),
                    probability=prob.item(),
                    logit=logits[0, idx].item()
                ))
        
        return {
            "input_text": text,
            "token_probabilities": token_probs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    """Export feature analysis data"""
    try:
        # Filter features by layer if specified
        features_to_export = {}
        for feature_id, feature_data in model_state.features.items():
            if layer_idx is None or feature_data["layer_idx"] == layer_idx:
                # Convert numpy arrays to lists for JSON serialization
                export_data = feature_data.copy()
                if isinstance(export_data["vector"], np.ndarray):
                    export_data["vector"] = export_data["vector"].tolist()
                features_to_export[feature_id] = export_data
        
        # Add metadata
        export_data = {
            "features": features_to_export,
            "metadata": {
                "model_name": model_state.model_name,
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "total_features": len(features_to_export),
                "layer_filter": layer_idx
            }
        }
        
        # Save to file
        output_file = Path(output_path)
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)
        
        return {
            "success": True,
            "output_file": str(output_file),
            "features_exported": len(features_to_export)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            return {
                "basic_info": feature_data,
                "detailed_analysis": detailed_analysis
            }
        except Exception as e:
            print(f"Warning: Could not get detailed analysis: {e}")
    
    return {"basic_info": feature_data}


# SAE Training Endpoints
@app.post("/sae/train")
async def train_sae(request: TrainSAERequest):
    """Start training a new SAE"""
    try:
        # Validate that we have a model loaded
        if model_state.model is None:
            raise HTTPException(status_code=400, detail="No model loaded. Load a model first before training SAE.")


        
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
                # Update status
                model_state.sae_training_jobs[job_id]["status"] = "preparing"
                
                # Check if activation data exists, if not, generate it
                if not Path(request.activation_data_path).exists():
                    print(f"Activation data not found at {request.activation_data_path}. Generating...")
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
                    print(f"Generated activation data at {request.activation_data_path}")
                
                # Validate that the parquet file is not empty
                if Path(request.activation_data_path).exists() and Path(request.activation_data_path).stat().st_size == 0:
                    raise Exception(f"Activation data file {request.activation_data_path} is empty. Please generate proper activation data first.")
                
                # Update status to training
                model_state.sae_training_jobs[job_id]["status"] = "training"
                
                # Create trainer
                trainer_module = SAETrainer(
                    activation_path=request.activation_data_path,
                    layer_idx=request.layer_idx,
                    latent_dim=request.latent_dim,
                    sparsity_coef=request.sparsity_coef,
                    learning_rate=request.learning_rate,
                    tied_weights=request.tied_weights,
                    activation_fn=request.activation_fn,
                    output_dir=str(output_path)
                )
                
                # Setup PyTorch Lightning trainer
                checkpoint_callback = ModelCheckpoint(
                    dirpath=output_path / "checkpoints",
                    filename="sae-{epoch:02d}-{train_loss:.4f}",
                    save_top_k=3,
                    monitor="train_loss",
                    mode="min"
                )
                
                trainer = pl.Trainer(
                    max_epochs=request.max_epochs,
                    callbacks=[checkpoint_callback],
                    enable_progress_bar=False,  # Disable for API usage
                    logger=False  # Disable logging for API usage
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
                
                # Note: WebSocket broadcast will be handled by polling mechanism
                
            except Exception as e:
                # Update job status with error
                model_state.sae_training_jobs[job_id].update({
                    "status": "failed",
                    "error": str(e),
                    "failed_at": pd.Timestamp.now().isoformat()
                })
                
                # Note: WebSocket broadcast will be handled by polling mechanism
        
        # Start training thread
        training_thread = threading.Thread(target=train_worker)
        training_thread.daemon = True
        training_thread.start()
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "SAE training started",
            "output_dir": str(output_path)
        }
        
    except Exception as e:
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
                    
                    # Add a small sparsity loss (placeholder)
                    sparsity_loss = current_loss * 0.1  # Rough estimate
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
    await manager.connect(websocket)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "inference_request":
                # Run inference and send results
                data = message.get("data", {})
                text = data.get("text") or message.get("text", "")  # Support both formats
                max_tokens = data.get("maxTokens", 50)
                
                result = await run_inference(text, max_tokens)
                
                response = {
                    "type": "inference_result",
                    "data": result
                }
                
                # Include requestId if provided for request-response pattern
                if "requestId" in message:
                    response["requestId"] = message["requestId"]
                
                await websocket.send_text(json.dumps(response))
            elif message["type"] == "ping":
                # Respond to ping with pong
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Note: Static file serving is handled by nginx in production
# This server only handles API endpoints and WebSocket connections


def main():
    """Main entry point for the server"""
    import uvicorn
    import os
    
    # Disable reload in production/systemd environment
    reload_enabled = os.environ.get("RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=reload_enabled,
        workers=1  # Single worker for systemd
    )


if __name__ == "__main__":
    main()