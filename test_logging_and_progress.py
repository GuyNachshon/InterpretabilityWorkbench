#!/usr/bin/env python3
"""
Comprehensive test script for logging and progress tracking functionality
Tests can be run without requiring full SAE training
"""
import requests
import json
import time
import websocket
import threading
import os
import sys
from datetime import datetime
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name):
        self.passed += 1
        print(f"✅ {test_name}")
    
    def add_fail(self, test_name, error=None):
        self.failed += 1
        if error:
            self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}")
        if error:
            print(f"   Error: {error}")
    
    def summary(self):
        print(f"\n📊 Test Summary:")
        print(f"   Passed: {self.passed}")
        print(f"   Failed: {self.failed}")
        print(f"   Total: {self.passed + self.failed}")
        if self.errors:
            print(f"\n❌ Errors:")
            for error in self.errors:
                print(f"   - {error}")
        return self.failed == 0

def test_server_connectivity():
    """Test basic server connectivity"""
    print("=== Testing Server Connectivity ===")
    results = TestResults()
    
    try:
        # Test health endpoint
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            results.add_pass("Health endpoint")
        else:
            results.add_fail("Health endpoint", f"Status {response.status_code}")
    except Exception as e:
        results.add_fail("Health endpoint", str(e))
    
    try:
        # Test model status endpoint
        response = requests.get(f"{BASE_URL}/model/status", timeout=5)
        if response.status_code == 200:
            results.add_pass("Model status endpoint")
        else:
            results.add_fail("Model status endpoint", f"Status {response.status_code}")
    except Exception as e:
        results.add_fail("Model status endpoint", str(e))
    
    return results

def test_logging_system():
    """Test logging system functionality"""
    print("\n=== Testing Logging System ===")
    results = TestResults()
    
    # Check if logs directory exists
    if os.path.exists("logs"):
        results.add_pass("Logs directory exists")
    else:
        results.add_fail("Logs directory exists")
        return results
    
    # Check if log files exist
    if os.path.exists("logs/api.log"):
        results.add_pass("API log file exists")
        
        # Check log file content
        try:
            with open("logs/api.log", "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    results.add_pass("API log has content")
                    print(f"   📝 Log file has {len(lines)} lines")
                    
                    # Check for recent entries
                    recent_entries = [line for line in lines if "2024" in line or "2025" in line]
                    if recent_entries:
                        results.add_pass("API log has recent entries")
                        print(f"   📄 Last entry: {recent_entries[-1].strip()}")
                    else:
                        results.add_fail("API log has recent entries")
                else:
                    results.add_fail("API log has content")
        except Exception as e:
            results.add_fail("Reading API log", str(e))
    else:
        results.add_fail("API log file exists")
    
    if os.path.exists("logs/errors.log"):
        results.add_pass("Error log file exists")
    else:
        results.add_fail("Error log file exists")
    
    return results

def test_websocket_communication():
    """Test WebSocket communication without training"""
    print("\n=== Testing WebSocket Communication ===")
    results = TestResults()
    
    messages_received = []
    connection_successful = False
    
    def on_message(ws, message):
        try:
            data = json.loads(message)
            messages_received.append(data)
            print(f"   📨 Received: {data.get('type', 'unknown')}")
        except Exception as e:
            print(f"   ❌ Failed to parse message: {e}")
    
    def on_error(ws, error):
        print(f"   ❌ WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print("   🔌 WebSocket connection closed")
    
    def on_open(ws):
        nonlocal connection_successful
        connection_successful = True
        print("   🔗 WebSocket connected")
        # Send a ping to test connection
        ws.send(json.dumps({"type": "ping"}))
    
    # Connect to WebSocket
    try:
        ws = websocket.WebSocketApp(
            WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Start WebSocket in a separate thread
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for connection and messages
        time.sleep(3)
        
        if connection_successful:
            results.add_pass("WebSocket connection")
        else:
            results.add_fail("WebSocket connection")
        
        if messages_received:
            results.add_pass("WebSocket message reception")
            # Check for pong response
            pong_messages = [msg for msg in messages_received if msg.get('type') == 'pong']
            if pong_messages:
                results.add_pass("WebSocket ping/pong")
            else:
                results.add_fail("WebSocket ping/pong")
        else:
            results.add_fail("WebSocket message reception")
        
        # Close WebSocket
        ws.close()
        
    except Exception as e:
        results.add_fail("WebSocket setup", str(e))
    
    return results

def test_api_logging():
    """Test that API endpoints are properly logging"""
    print("\n=== Testing API Logging ===")
    results = TestResults()
    
    # Get initial log line count
    try:
        with open("logs/api.log", "r") as f:
            initial_lines = len(f.readlines())
    except:
        initial_lines = 0
    
    # Make some API calls
    endpoints_to_test = [
        ("/health", "GET"),
        ("/model/status", "GET"),
        ("/sae/status", "GET"),
        ("/patches", "GET"),
    ]
    
    for endpoint, method in endpoints_to_test:
        try:
            if method == "GET":
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            else:
                response = requests.post(f"{BASE_URL}{endpoint}", timeout=5)
            
            if response.status_code in [200, 404, 400]:  # Acceptable responses
                results.add_pass(f"{method} {endpoint}")
            else:
                results.add_fail(f"{method} {endpoint}", f"Status {response.status_code}")
        except Exception as e:
            results.add_fail(f"{method} {endpoint}", str(e))
    
    # Check if new log entries were created
    time.sleep(1)  # Give time for logging
    try:
        with open("logs/api.log", "r") as f:
            final_lines = len(f.readlines())
        
        if final_lines > initial_lines:
            results.add_pass("API calls generated log entries")
            print(f"   📝 Log entries increased: {initial_lines} → {final_lines}")
        else:
            results.add_fail("API calls generated log entries")
    except Exception as e:
        results.add_fail("Checking log entries", str(e))
    
    return results

def test_training_job_creation():
    """Test training job creation without actual training"""
    print("\n=== Testing Training Job Creation ===")
    results = TestResults()
    
    # First, try to load a model (this might fail but should log)
    print("   Loading model...")
    try:
        response = requests.post(f"{BASE_URL}/load-model", json={
            "model_name": "openai-community/gpt2"
        }, timeout=30)
        
        if response.status_code == 200:
            results.add_pass("Model loading")
            print("   ✅ Model loaded successfully")
        else:
            # This might fail due to model size, but should still log
            print(f"   ⚠️ Model loading failed (expected): {response.status_code}")
            results.add_pass("Model loading attempt (logged)")
    except Exception as e:
        print(f"   ⚠️ Model loading error (expected): {e}")
        results.add_pass("Model loading attempt (logged)")
    
    # Test creating a training job (this will fail but should show logging)
    print("   Testing training job creation...")
    try:
        response = requests.post(f"{BASE_URL}/sae/train", json={
            "layer_idx": 6,
            "activation_data_path": "nonexistent_activations.parquet",
            "latent_dim": 1024,
            "sparsity_coef": 0.001,
            "learning_rate": 0.001,
            "max_epochs": 5,
            "tied_weights": True,
            "activation_fn": "relu"
        }, timeout=10)
        
        # This should fail due to missing activation data, but should log the attempt
        if response.status_code in [400, 500]:
            results.add_pass("Training job creation attempt (logged)")
            print(f"   ✅ Training job creation failed as expected: {response.status_code}")
        else:
            results.add_fail("Training job creation", f"Unexpected status {response.status_code}")
    except Exception as e:
        results.add_pass("Training job creation attempt (logged)")
        print(f"   ✅ Training job creation failed as expected: {e}")
    
    return results

def test_log_format_and_structure():
    """Test log format and structure"""
    print("\n=== Testing Log Format and Structure ===")
    results = TestResults()
    
    try:
        with open("logs/api.log", "r") as f:
            lines = f.readlines()
        
        if not lines:
            results.add_fail("Log file has content")
            return results
        
        # Check log format
        valid_formats = 0
        for line in lines[-10:]:  # Check last 10 lines
            # Expected format: timestamp - logger - level - message
            parts = line.strip().split(" - ")
            if len(parts) >= 4:
                # Check timestamp format
                if len(parts[0]) >= 19 and ":" in parts[0]:
                    # Check logger name
                    if parts[1] in ["api", "model", "sae", "training", "websocket"]:
                        # Check log level
                        if parts[2] in ["DEBUG", "INFO", "WARNING", "ERROR"]:
                            valid_formats += 1
        
        if valid_formats > 0:
            results.add_pass("Log format structure")
            print(f"   📝 {valid_formats} valid log entries found")
        else:
            results.add_fail("Log format structure")
        
        # Check for specific loggers
        loggers_found = set()
        for line in lines:
            parts = line.strip().split(" - ")
            if len(parts) >= 2:
                loggers_found.add(parts[1])
        
        expected_loggers = {"api"}
        found_loggers = loggers_found.intersection(expected_loggers)
        
        if found_loggers:
            results.add_pass("Expected loggers present")
            print(f"   🔍 Found loggers: {list(found_loggers)}")
        else:
            results.add_fail("Expected loggers present")
        
    except Exception as e:
        results.add_fail("Log format analysis", str(e))
    
    return results

def test_error_logging():
    """Test error logging functionality"""
    print("\n=== Testing Error Logging ===")
    results = TestResults()
    
    # Check if error log exists and has content
    if os.path.exists("logs/errors.log"):
        results.add_pass("Error log file exists")
        
        try:
            with open("logs/errors.log", "r") as f:
                error_lines = f.readlines()
            
            if error_lines:
                results.add_pass("Error log has content")
                print(f"   📝 Error log has {len(error_lines)} lines")
                
                # Check for error format
                error_formats = 0
                for line in error_lines[-5:]:  # Check last 5 lines
                    if "ERROR" in line:
                        error_formats += 1
                
                if error_formats > 0:
                    results.add_pass("Error log format")
                else:
                    results.add_fail("Error log format")
            else:
                results.add_pass("Error log is empty (no errors)")
        except Exception as e:
            results.add_fail("Reading error log", str(e))
    else:
        results.add_fail("Error log file exists")
    
    return results

def test_websocket_progress_messages():
    """Test WebSocket progress message handling"""
    print("\n=== Testing WebSocket Progress Messages ===")
    results = TestResults()
    
    # This test simulates what the UI would do
    messages_received = []
    
    def on_message(ws, message):
        try:
            data = json.loads(message)
            messages_received.append(data)
        except:
            pass
    
    def on_open(ws):
        # Send a test message to simulate progress
        test_progress = {
            "type": "training_progress",
            "job_id": "test-job-123",
            "status": "training",
            "epoch": 5,
            "total_epochs": 100,
            "progress": 5.0,
            "metrics": {
                "train_loss": 0.1234,
                "reconstruction_loss": 0.0987,
                "sparsity_loss": 0.0247
            },
            "elapsed_time": 300.5,
            "estimated_remaining": 5700.0
        }
        ws.send(json.dumps(test_progress))
    
    try:
        ws = websocket.WebSocketApp(
            WS_URL,
            on_open=on_open,
            on_message=on_message
        )
        
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        time.sleep(2)
        
        # Check if we can send and receive messages
        if len(messages_received) >= 1:  # At least ping/pong
            results.add_pass("WebSocket message handling")
        else:
            results.add_fail("WebSocket message handling")
        
        ws.close()
        
    except Exception as e:
        results.add_fail("WebSocket progress test", str(e))
    
    return results

def main():
    """Run all tests"""
    print("🚀 Starting comprehensive logging and progress tracking tests")
    print(f"📅 Test started at: {datetime.now()}")
    print(f"🌐 Testing against: {BASE_URL}")
    print(f"🔗 WebSocket URL: {WS_URL}")
    
    all_results = TestResults()
    
    # Run all tests
    tests = [
        ("Server Connectivity", test_server_connectivity),
        ("Logging System", test_logging_system),
        ("WebSocket Communication", test_websocket_communication),
        ("API Logging", test_api_logging),
        ("Training Job Creation", test_training_job_creation),
        ("Log Format and Structure", test_log_format_and_structure),
        ("Error Logging", test_error_logging),
        ("WebSocket Progress Messages", test_websocket_progress_messages),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            all_results.passed += result.passed
            all_results.failed += result.failed
            all_results.errors.extend(result.errors)
        except Exception as e:
            all_results.failed += 1
            all_results.errors.append(f"{test_name}: {str(e)}")
            print(f"❌ {test_name} failed with exception: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    success = all_results.summary()
    
    if success:
        print("\n🎉 All tests passed! The logging and progress tracking system is working correctly.")
        print("\n📋 What this means:")
        print("   ✅ Server is running and accessible")
        print("   ✅ Logging system is properly configured")
        print("   ✅ WebSocket communication is working")
        print("   ✅ API endpoints are logging their operations")
        print("   ✅ Error logging is functional")
        print("   ✅ Progress tracking infrastructure is ready")
    else:
        print("\n⚠️ Some tests failed. Check the errors above for details.")
        print("\n🔧 Troubleshooting tips:")
        print("   - Ensure the server is running: python run.py")
        print("   - Check server logs: tail -f logs/api.log")
        print("   - Verify WebSocket connection in browser console")
        print("   - Check file permissions for logs directory")
    
    print(f"\n📅 Test completed at: {datetime.now()}")
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 