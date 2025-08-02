"""
WebSocket handlers for real-time model interaction
"""
import json
import asyncio
import time
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import torch
import numpy as np
from datetime import datetime


class WebSocketManager:
    """Manages WebSocket connections and real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "subscriptions": set()
        }
        
        await self.send_to_client(client_id, {
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": time.time()
        })
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_metadata:
            del self.connection_metadata[client_id]
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
                self.connection_metadata[client_id]["last_activity"] = datetime.now()
                return True
            except Exception as e:
                print(f"Error sending to client {client_id}: {e}")
                self.disconnect(client_id)
                return False
        return False
    
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[List[str]] = None):
        """Broadcast message to all connected clients"""
        exclude = exclude or []
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            if client_id not in exclude:
                try:
                    await websocket.send_text(json.dumps(message))
                    self.connection_metadata[client_id]["last_activity"] = datetime.now()
                except Exception as e:
                    print(f"Error broadcasting to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def subscribe_client(self, client_id: str, subscription_type: str):
        """Subscribe client to specific message types"""
        if client_id in self.connection_metadata:
            self.connection_metadata[client_id]["subscriptions"].add(subscription_type)
    
    async def unsubscribe_client(self, client_id: str, subscription_type: str):
        """Unsubscribe client from specific message types"""
        if client_id in self.connection_metadata:
            self.connection_metadata[client_id]["subscriptions"].discard(subscription_type)
    
    def get_subscribers(self, subscription_type: str) -> List[str]:
        """Get list of clients subscribed to a message type"""
        subscribers = []
        for client_id, metadata in self.connection_metadata.items():
            if subscription_type in metadata["subscriptions"]:
                subscribers.append(client_id)
        return subscribers


class ModelInteractionHandler:
    """Handles real-time model interactions via WebSocket"""
    
    def __init__(self, model_state, ws_manager: WebSocketManager):
        self.model_state = model_state
        self.ws_manager = ws_manager
        
    async def handle_inference_request(self, client_id: str, message: Dict[str, Any]):
        """Handle real-time inference request"""
        start_time = time.time()
        
        try:
            text = message.get("text", "")
            max_tokens = message.get("max_tokens", 20)
            temperature = message.get("temperature", 1.0)
            
            if not self.model_state.model or not self.model_state.tokenizer:
                await self.ws_manager.send_to_client(client_id, {
                    "type": "error",
                    "message": "No model loaded",
                    "request_id": message.get("request_id")
                })
                return
            
            # Tokenize input
            inputs = self.model_state.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            device = next(self.model_state.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model_state.model(**inputs)
                
                # Get logits for next token prediction
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits[:, -1, :]
                else:
                    # For encoder-only models, we might need to add a language modeling head
                    # For now, just use the last hidden state
                    hidden_states = outputs.last_hidden_state[:, -1, :]
                    logits = hidden_states  # This is a simplification
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, k=50)
                
                # Prepare response data
                token_data = []
                for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
                    token = self.model_state.tokenizer.decode([idx.item()])
                    token_data.append({
                        "token": token,
                        "token_id": idx.item(),
                        "probability": prob.item(),
                        "logit": logits[0, idx].item()
                    })
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            # Send response
            await self.ws_manager.send_to_client(client_id, {
                "type": "inference_result",
                "data": {
                    "input_text": text,
                    "token_probabilities": token_data,
                    "latency_ms": latency,
                    "model_name": self.model_state.model_name
                },
                "request_id": message.get("request_id"),
                "timestamp": time.time()
            })
            
        except Exception as e:
            await self.ws_manager.send_to_client(client_id, {
                "type": "error",
                "message": str(e),
                "request_id": message.get("request_id")
            })
    
    async def handle_patch_toggle(self, client_id: str, message: Dict[str, Any]):
        """Handle patch toggle request"""
        try:
            patch_id = message.get("patch_id")
            if not patch_id:
                raise ValueError("patch_id is required")
            
            if not self.model_state.patcher:
                raise ValueError("No model loaded for patching")
            
            patches = self.model_state.patcher.list_patches()
            if patch_id not in patches:
                raise ValueError(f"Patch {patch_id} not found")
            
            # Toggle patch
            if patches[patch_id]["enabled"]:
                self.model_state.patcher.disable_patch(patch_id)
                action = "disabled"
            else:
                self.model_state.patcher.enable_patch(patch_id)
                action = "enabled"
            
            # Broadcast to all subscribers
            await self.ws_manager.broadcast({
                "type": "patch_toggled",
                "patch_id": patch_id,
                "action": action,
                "timestamp": time.time()
            })
            
        except Exception as e:
            await self.ws_manager.send_to_client(client_id, {
                "type": "error",
                "message": str(e),
                "request_id": message.get("request_id")
            })
    
    async def handle_feature_analysis(self, client_id: str, message: Dict[str, Any]):
        """Handle feature analysis request"""
        try:
            feature_id = message.get("feature_id")
            if not feature_id or feature_id not in self.model_state.features:
                raise ValueError(f"Feature {feature_id} not found")
            
            feature_data = self.model_state.features[feature_id]
            
            # Send detailed feature analysis
            await self.ws_manager.send_to_client(client_id, {
                "type": "feature_analysis",
                "data": {
                    "feature_id": feature_id,
                    "layer_idx": feature_data["layer_idx"],
                    "sparsity": feature_data["sparsity"],
                    "activation_strength": feature_data["activation_strength"],
                    "vector_norm": float(np.linalg.norm(feature_data["vector"])),
                    "top_activating_dimensions": self._get_top_dimensions(feature_data["vector"])
                },
                "request_id": message.get("request_id"),
                "timestamp": time.time()
            })
            
        except Exception as e:
            await self.ws_manager.send_to_client(client_id, {
                "type": "error",
                "message": str(e),
                "request_id": message.get("request_id")
            })
    
    def _get_top_dimensions(self, vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top-k dimensions by absolute value"""
        abs_vector = np.abs(vector)
        top_indices = np.argsort(abs_vector)[-top_k:][::-1]
        
        return [
            {
                "dimension": int(idx),
                "value": float(vector[idx]),
                "abs_value": float(abs_vector[idx])
            }
            for idx in top_indices
        ]
    
    async def handle_subscription(self, client_id: str, message: Dict[str, Any]):
        """Handle subscription requests"""
        subscription_type = message.get("subscription_type")
        action = message.get("action", "subscribe")
        
        if action == "subscribe":
            await self.ws_manager.subscribe_client(client_id, subscription_type)
        elif action == "unsubscribe":
            await self.ws_manager.unsubscribe_client(client_id, subscription_type)
        
        await self.ws_manager.send_to_client(client_id, {
            "type": "subscription_response",
            "subscription_type": subscription_type,
            "action": action,
            "success": True
        })


async def handle_websocket_message(
    client_id: str,
    message: Dict[str, Any],
    model_state,
    ws_manager: WebSocketManager
):
    """Main message handler for WebSocket connections"""
    
    handler = ModelInteractionHandler(model_state, ws_manager)
    message_type = message.get("type")
    
    if message_type == "inference_request":
        await handler.handle_inference_request(client_id, message)
    elif message_type == "patch_toggle":
        await handler.handle_patch_toggle(client_id, message)
    elif message_type == "feature_analysis":
        await handler.handle_feature_analysis(client_id, message)
    elif message_type == "subscription":
        await handler.handle_subscription(client_id, message)
    elif message_type == "ping":
        await ws_manager.send_to_client(client_id, {
            "type": "pong",
            "timestamp": time.time()
        })
    else:
        await ws_manager.send_to_client(client_id, {
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        })


async def cleanup_inactive_connections(ws_manager: WebSocketManager, timeout_minutes: int = 30):
    """Periodically clean up inactive connections"""
    while True:
        await asyncio.sleep(60)  # Check every minute
        
        current_time = datetime.now()
        inactive_clients = []
        
        for client_id, metadata in ws_manager.connection_metadata.items():
            time_since_activity = current_time - metadata["last_activity"]
            if time_since_activity.total_seconds() > (timeout_minutes * 60):
                inactive_clients.append(client_id)
        
        for client_id in inactive_clients:
            print(f"Cleaning up inactive client: {client_id}")
            ws_manager.disconnect(client_id)