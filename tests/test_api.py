#!/usr/bin/env python3
"""
Comprehensive API tests for InterpretabilityWorkbench
"""
import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from pathlib import Path
import torch
import numpy as np

# Import the app
import sys
sys.path.append(str(Path(__file__).parent.parent))
from interpretability_workbench.server.api import app, model_state

client = TestClient(app)

@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = Mock()
    model.return_value.logits = torch.randn(1, 10, 50257)  # GPT-2 vocab size
    model.return_value.last_hidden_state = torch.randn(1, 10, 768)
    return model

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing"""
    tokenizer = Mock()
    tokenizer.decode.return_value = "test_token"
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|endoftext|>"
    return tokenizer

@pytest.fixture
def mock_sae():
    """Create a mock SAE for testing"""
    sae = Mock()
    sae.encoder = torch.nn.Linear(768, 16384)
    sae.decoder = torch.nn.Linear(16384, 768)
    sae.input_dim = 768
    sae.latent_dim = 16384
    sae.tied_weights = True
    return sae

@pytest.fixture
def setup_mock_model(mock_model, mock_tokenizer):
    """Setup mock model in model_state"""
    model_state.model = mock_model
    model_state.tokenizer = mock_tokenizer
    model_state.model_name = "test-model"
    yield
    # Cleanup
    model_state.model = None
    model_state.tokenizer = None
    model_state.model_name = None

class TestHealthEndpoints:
    """Test health and status endpoints"""
    
    def test_health_check(self):
        """Test basic health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_model_status_no_model(self):
        """Test model status when no model is loaded"""
        response = client.get("/model/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"
    
    def test_model_status_with_model(self, setup_mock_model):
        """Test model status when model is loaded"""
        response = client.get("/model/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["model_name"] == "test-model"
    
    def test_sae_status_no_sae(self):
        """Test SAE status when no SAE is loaded"""
        response = client.get("/sae/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"
    
    def test_sae_status_with_sae(self, mock_sae):
        """Test SAE status when SAE is loaded"""
        model_state.sae_models[6] = mock_sae
        model_state.features = {"test_feature": {"id": "test"}}
        
        response = client.get("/sae/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["layerCount"] == 1
        assert data["featureCount"] == 1
        
        # Cleanup
        del model_state.sae_models[6]
        model_state.features = {}

class TestModelEndpoints:
    """Test model loading endpoints"""
    
    @patch('interpretability_workbench.server.api.AutoModel')
    @patch('interpretability_workbench.server.api.AutoTokenizer')
    def test_load_model_success(self, mock_auto_tokenizer, mock_auto_model):
        """Test successful model loading"""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model
        
        response = client.post("/load-model", json={"model_name": "distilgpt2"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["success"] is True
    
    def test_load_model_invalid_name(self):
        """Test model loading with invalid model name"""
        response = client.post("/load-model", json={"model_name": ""})
        assert response.status_code == 400
    
    def test_load_model_missing_field(self):
        """Test model loading with missing field"""
        response = client.post("/load-model", json={})
        assert response.status_code == 422

class TestInferenceEndpoints:
    """Test inference endpoints"""
    
    def test_inference_no_model(self):
        """Test inference without model loaded"""
        response = client.post("/inference", json={"text": "Hello world"})
        assert response.status_code == 400
        assert "No model loaded" in response.json()["detail"]
    
    def test_inference_empty_text(self, setup_mock_model):
        """Test inference with empty text"""
        response = client.post("/inference", json={"text": ""})
        assert response.status_code == 400
        assert "Text cannot be empty" in response.json()["detail"]
    
    def test_inference_success(self, setup_mock_model):
        """Test successful inference"""
        response = client.post("/inference", json={"text": "Hello world"})
        assert response.status_code == 200
        data = response.json()
        assert "original" in data
        assert "patched" in data
        assert len(data["original"]) > 0
        assert len(data["patched"]) > 0
    
    def test_inference_caching(self, setup_mock_model):
        """Test inference caching"""
        # First request
        response1 = client.post("/inference", json={"text": "Test caching"})
        assert response1.status_code == 200
        
        # Second request (should be cached)
        response2 = client.post("/inference", json={"text": "Test caching"})
        assert response2.status_code == 200
        
        # Results should be identical
        assert response1.json() == response2.json()


class TestCacheEndpoints:
    """Test cache-related endpoints"""
    
    def test_cache_stats(self):
        """Test getting cache statistics"""
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "cache_stats" in data
    
    def test_clear_cache(self):
        """Test clearing cache"""
        response = client.post("/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

class TestSAEEndpoints:
    """Test SAE-related endpoints"""
    
    def test_train_sae_missing_file(self):
        """Test SAE training with missing activation file"""
        response = client.post("/sae/train", json={
            "layer_idx": 6,
            "activation_data_path": "nonexistent_file.parquet",
            "latent_dim": 16384
        })
        assert response.status_code == 500
    
    def test_train_sae_invalid_params(self):
        """Test SAE training with invalid parameters"""
        response = client.post("/sae/train", json={
            "layer_idx": -1,  # Invalid layer index
            "activation_data_path": "test.parquet",
            "latent_dim": 0  # Invalid latent dim
        })
        assert response.status_code == 422
    
    def test_training_jobs_empty(self):
        """Test getting training jobs when none exist"""
        response = client.get("/sae/training/jobs")
        assert response.status_code == 200
        data = response.json()
        assert data["jobs"] == []
        assert data["total"] == 0
    
    def test_training_status_not_found(self):
        """Test getting status of non-existent training job"""
        response = client.get("/sae/training/status/nonexistent-job")
        assert response.status_code == 404
    
    def test_cancel_training_not_found(self):
        """Test cancelling non-existent training job"""
        response = client.delete("/sae/training/nonexistent-job")
        assert response.status_code == 404

class TestFeatureEndpoints:
    """Test feature-related endpoints"""
    
    def test_get_features_no_sae(self):
        """Test getting features without SAE loaded"""
        response = client.get("/features")
        assert response.status_code == 200
        data = response.json()
        assert data["features"] == []
        assert data["total"] == 0
    
    def test_get_features_with_sae(self, mock_sae):
        """Test getting features with SAE loaded"""
        # Setup mock features
        model_state.sae_models[6] = mock_sae
        model_state.features = {
            "layer_6_feature_0": {
                "id": "layer_6_feature_0",
                "layer_idx": 6,
                "feature_idx": 0,
                "sparsity": 0.5,
                "activation_strength": 1.0,
                "top_tokens": [{"token": "test", "strength": 0.8}]
            }
        }
        
        response = client.get("/features")
        assert response.status_code == 200
        data = response.json()
        assert len(data["features"]) == 1
        assert data["total"] == 1
        
        # Cleanup
        del model_state.sae_models[6]
        model_state.features = {}
    
    def test_get_feature_details_not_found(self):
        """Test getting details of non-existent feature"""
        response = client.get("/feature/nonexistent-feature/details")
        assert response.status_code == 404

class TestPatchEndpoints:
    """Test patch-related endpoints"""
    
    def test_create_patch_no_model(self):
        """Test creating patch without model loaded"""
        response = client.post("/patch", json={
            "feature_id": "test_feature",
            "layer_idx": 6,
            "strength": 1.0,
            "enabled": True
        })
        assert response.status_code == 400
    
    def test_get_patches_no_model(self):
        """Test getting patches without model loaded"""
        response = client.get("/patches")
        assert response.status_code == 200
        data = response.json()
        assert data["patches"] == []
    
    def test_update_patch_not_found(self):
        """Test updating non-existent patch"""
        response = client.patch("/patch/nonexistent-patch", json={"strength": 1.0})
        assert response.status_code == 404
    
    def test_toggle_patch_not_found(self):
        """Test toggling non-existent patch"""
        response = client.post("/patch/nonexistent-patch/toggle")
        assert response.status_code == 404

class TestExportEndpoints:
    """Test export endpoints"""
    
    def test_export_sae_no_sae(self):
        """Test exporting SAE when none loaded"""
        response = client.post("/export-sae", json={
            "layer_idx": 6,
            "output_path": "test_output"
        })
        assert response.status_code == 404
    
    def test_export_patches_no_model(self):
        """Test exporting patches without model loaded"""
        response = client.post("/export-patches", json={"output_path": "test_output"})
        assert response.status_code == 400
    
    def test_export_features_no_features(self):
        """Test exporting features when none exist"""
        response = client.post("/export-features", json={"output_path": "test_output"})
        assert response.status_code == 200
        # Should return empty export

class TestWebSocketEndpoints:
    """Test WebSocket endpoints"""
    
    def test_websocket_connection(self):
        """Test WebSocket connection"""
        with client.websocket_connect("/ws") as websocket:
            # Send a test message
            websocket.send_text(json.dumps({"type": "ping"}))
            # Should receive a response
            data = websocket.receive_text()
            assert data is not None

class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_json(self):
        """Test handling of invalid JSON"""
        response = client.post("/load-model", data="invalid json")
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        response = client.post("/load-model", json={"wrong_field": "value"})
        assert response.status_code == 422
    
    def test_large_text_input(self, setup_mock_model):
        """Test handling of very large text input"""
        large_text = "a" * 10000  # Very large text
        response = client.post("/inference", json={"text": large_text})
        # Should handle gracefully (might truncate or error appropriately)
        assert response.status_code in [200, 400, 413]

class TestPerformance:
    """Test performance characteristics"""
    
    def test_health_check_performance(self):
        """Test health check response time"""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 0.1  # Should respond in < 100ms
    
    def test_model_status_performance(self):
        """Test model status response time"""
        start_time = time.time()
        response = client.get("/model/status")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 0.1  # Should respond in < 100ms

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 