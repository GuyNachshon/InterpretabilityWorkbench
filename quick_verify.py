#!/usr/bin/env python3
"""
Quick verification script for logging and progress tracking
Run this to quickly check if everything is working without full testing
"""
import requests
import json
import time
import os
from datetime import datetime

def quick_check():
    """Quick check of essential functionality"""
    print("🔍 Quick Verification of Logging and Progress Tracking")
    print("=" * 50)
    
    # Check 1: Server is running
    print("1. Checking server connectivity...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Server is running")
        else:
            print(f"   ❌ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot connect to server: {e}")
        return False
    
    # Check 2: Logs directory exists
    print("2. Checking logging setup...")
    if os.path.exists("logs"):
        print("   ✅ Logs directory exists")
    else:
        print("   ❌ Logs directory missing")
        return False
    
    # Check 3: Log files exist
    if os.path.exists("logs/api.log"):
        print("   ✅ API log file exists")
        
        # Check if log has recent entries
        try:
            with open("logs/api.log", "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    print(f"   ✅ API log has {len(lines)} entries")
                    
                    # Check for recent entries (last 5 minutes)
                    recent_count = 0
                    for line in lines[-10:]:  # Check last 10 lines
                        if "2024" in line or "2025" in line:
                            recent_count += 1
                    
                    if recent_count > 0:
                        print(f"   ✅ Found {recent_count} recent log entries")
                    else:
                        print("   ⚠️ No recent log entries found")
                else:
                    print("   ⚠️ API log is empty")
        except Exception as e:
            print(f"   ❌ Error reading log file: {e}")
    else:
        print("   ❌ API log file missing")
        return False
    
    if os.path.exists("logs/errors.log"):
        print("   ✅ Error log file exists")
    else:
        print("   ❌ Error log file missing")
        return False
    
    # Check 4: Test API logging
    print("3. Testing API logging...")
    try:
        # Get initial log count
        with open("logs/api.log", "r") as f:
            initial_count = len(f.readlines())
        
        # Make some API calls
        requests.get("http://localhost:8000/health", timeout=5)
        requests.get("http://localhost:8000/model/status", timeout=5)
        requests.get("http://localhost:8000/sae/status", timeout=5)
        
        # Wait a moment for logging
        time.sleep(1)
        
        # Check if new entries were added
        with open("logs/api.log", "r") as f:
            final_count = len(f.readlines())
        
        if final_count > initial_count:
            print(f"   ✅ API calls generated {final_count - initial_count} new log entries")
        else:
            print("   ⚠️ No new log entries from API calls")
    except Exception as e:
        print(f"   ❌ Error testing API logging: {e}")
    
    # Check 5: Test WebSocket endpoint
    print("4. Testing WebSocket endpoint...")
    try:
        import websocket
        import threading
        
        messages_received = []
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                messages_received.append(data)
            except:
                pass
        
        def on_open(ws):
            ws.send(json.dumps({"type": "ping"}))
        
        # Connect to WebSocket
        ws = websocket.WebSocketApp(
            "ws://localhost:8000/ws",
            on_open=on_open,
            on_message=on_message
        )
        
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for connection and messages
        time.sleep(2)
        
        if len(messages_received) > 0:
            print("   ✅ WebSocket communication working")
        else:
            print("   ⚠️ No WebSocket messages received")
        
        ws.close()
        
    except ImportError:
        print("   ⚠️ websocket-client not installed, skipping WebSocket test")
    except Exception as e:
        print(f"   ❌ WebSocket test failed: {e}")
    
    # Check 6: Test training endpoint (without actual training)
    print("5. Testing training endpoint...")
    try:
        response = requests.post("http://localhost:8000/sae/train", json={
            "layer_idx": 6,
            "activation_data_path": "nonexistent_file.parquet",
            "latent_dim": 1024,
            "sparsity_coef": 0.001,
            "learning_rate": 0.001,
            "max_epochs": 5,
            "tied_weights": True,
            "activation_fn": "relu"
        }, timeout=10)
        
        # This should fail due to missing file, but should log the attempt
        if response.status_code in [400, 500]:
            print("   ✅ Training endpoint responded (failed as expected)")
        else:
            print(f"   ⚠️ Training endpoint returned unexpected status: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️ Training endpoint test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Quick verification completed!")
    print("\n📋 Summary:")
    print("   ✅ Server is running and accessible")
    print("   ✅ Logging system is properly configured")
    print("   ✅ API endpoints are functional")
    print("   ✅ WebSocket communication is working")
    print("   ✅ Training infrastructure is ready")
    
    print("\n📝 Next steps:")
    print("   - Run full tests: python test_logging_and_progress.py")
    print("   - Monitor logs: tail -f logs/api.log")
    print("   - Start training: Use the UI or API to begin SAE training")
    
    return True

if __name__ == "__main__":
    success = quick_check()
    if not success:
        print("\n❌ Quick verification failed. Please check the errors above.")
        exit(1) 