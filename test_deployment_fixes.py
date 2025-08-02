#!/usr/bin/env python3
"""
Test script to verify deployment fixes
"""
import requests
import json
import time

def test_api_endpoints():
    """Test the main API endpoints"""
    # Test both direct backend and nginx proxy
    base_urls = ["http://localhost:8000", "http://localhost"]
    
    for base_url in base_urls:
        print(f"\nTesting {base_url}...")
    
    print("Testing API endpoints...")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test model status
    try:
        response = requests.get(f"{base_url}/model/status")
        print(f"Model status: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Model status failed: {e}")
    
    # Test SAE status
    try:
        response = requests.get(f"{base_url}/sae/status")
        print(f"SAE status: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"SAE status failed: {e}")
    
    # Test patches endpoint
    try:
        response = requests.get(f"{base_url}/patches")
        print(f"Patches: {response.status_code} - {response.json()}")
        # Verify it returns a list
        patches = response.json()
        if isinstance(patches, list):
            print("✓ Patches endpoint returns a list (correct format)")
        else:
            print("✗ Patches endpoint does not return a list")
    except Exception as e:
        print(f"Patches endpoint failed: {e}")

def test_websocket_connection():
    """Test WebSocket connection"""
    import websocket
    
    print("\nTesting WebSocket connection...")
    
    try:
        # Test WebSocket connection
        ws = websocket.create_connection("ws://localhost:8000/ws", timeout=5)
        print("✓ WebSocket connection successful")
        
        # Send a ping
        ws.send(json.dumps({"type": "ping"}))
        response = ws.recv()
        print(f"Ping response: {response}")
        
        ws.close()
    except Exception as e:
        print(f"✗ WebSocket connection failed: {e}")

if __name__ == "__main__":
    print("Testing deployment fixes...")
    test_api_endpoints()
    test_websocket_connection()
    print("\nTest completed!")
    
    print("\nFor nginx deployment:")
    print("1. Update nginx.conf with your domain and paths")
    print("2. Copy nginx.conf to /etc/nginx/sites-available/")
    print("3. Enable the site and reload nginx")
    print("4. Start the FastAPI backend with: python run.py") 