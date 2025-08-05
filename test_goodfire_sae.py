#!/usr/bin/env python3
"""
Test script for Goodfire SAE integration
Demonstrates how to load and use pre-trained SAEs from Goodfire
"""

import requests
import json
import time

# Configuration
MODEL_NAME = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
SAE_NAME = 'Llama-3.1-8B-Instruct-SAE-l19'
BASE_URL = 'http://localhost:8000'  # Direct backend URL

def test_goodfire_sae_integration():
    """Test the Goodfire SAE integration"""
    
    print("🚀 Testing Goodfire SAE Integration")
    print("=" * 50)
    
    # Step 1: Load the model
    print(f"\n1. Loading model: {MODEL_NAME}")
    response = requests.post(f"{BASE_URL}/load-model", json={
        "model_name": MODEL_NAME
    })
    
    if response.status_code != 200:
        print(f"❌ Failed to load model: {response.text}")
        return
    
    print("✅ Model loaded successfully")
    
    # Step 2: Load the Goodfire SAE (converted to standard format)
    print(f"\n2. Loading Goodfire SAE: {SAE_NAME}")
    print("Note: First run: python -m interpretability_workbench.goodfire_converter")
    
    # Use the converted safetensors file
    sae_path = "sae_layer_19.safetensors"  # Converted file path
    
    response = requests.post(f"{BASE_URL}/load-sae", json={
        "layer_idx": 19,  # Layer 19 for Llama-3.1-8B-Instruct-SAE-l19
        "saePath": sae_path,
        "activationsPath": None  # Optional: path to activation data if you have it
    })
    
    if response.status_code != 200:
        print(f"❌ Failed to load SAE: {response.text}")
        return
    
    sae_info = response.json()
    print(f"✅ SAE loaded successfully:")
    print(f"   - Layer index: {sae_info['layer_idx']}")
    print(f"   - Layer name: {sae_info['layer_name']}")
    print(f"   - Expansion factor: {sae_info['expansion_factor']}")
    
    # Step 3: Check SAE status
    print(f"\n3. Checking SAE status")
    response = requests.get(f"{BASE_URL}/sae/status")
    
    if response.status_code == 200:
        status = response.json()
        print(f"✅ SAE Status: {status}")
    else:
        print(f"⚠️ Could not get SAE status: {response.text}")
    
    # Step 4: Get features (if available)
    print(f"\n4. Getting features")
    response = requests.get(f"{BASE_URL}/features", params={
        "layer_idx": sae_info['layer_idx'],
        "limit": 10
    })
    
    if response.status_code == 200:
        features = response.json()
        print(f"✅ Found {features['total']} features")
        if features['features']:
            print("   Top features:")
            for i, feature in enumerate(features['features'][:5]):
                print(f"   {i+1}. {feature['id']} (activation: {feature['activation_strength']:.4f})")
    else:
        print(f"⚠️ Could not get features: {response.text}")
    
    # Step 5: Test inference
    print(f"\n5. Testing inference")
    response = requests.post(f"{BASE_URL}/inference", json={
        "text": "Hello, how are you?",
        "max_length": 20
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Inference successful:")
        print(f"   Input: {result['inputText']}")
        print(f"   Top tokens:")
        for i, token in enumerate(result['tokenProbabilities'][:5]):
            print(f"   {i+1}. '{token['token']}' (prob: {token['probability']:.4f})")
    else:
        print(f"⚠️ Could not run inference: {response.text}")
    
    print(f"\n🎉 Goodfire SAE integration test completed!")

if __name__ == "__main__":
    test_goodfire_sae_integration() 