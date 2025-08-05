#!/usr/bin/env python3
"""
Real-time log monitoring script
Watch logs as they happen to verify logging is working
"""
import time
import os
import sys
from datetime import datetime
from pathlib import Path

def monitor_logs():
    """Monitor logs in real-time"""
    print("📊 Real-time Log Monitor")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    # Check if logs directory exists
    if not os.path.exists("logs"):
        print("❌ Logs directory not found. Start the server first.")
        return
    
    # Check if log files exist
    api_log_path = "logs/api.log"
    error_log_path = "logs/errors.log"
    
    if not os.path.exists(api_log_path):
        print("❌ API log file not found. Start the server first.")
        return
    
    print(f"📝 Monitoring: {api_log_path}")
    if os.path.exists(error_log_path):
        print(f"⚠️  Also monitoring: {error_log_path}")
    print()
    
    # Get initial file sizes
    api_size = os.path.getsize(api_log_path) if os.path.exists(api_log_path) else 0
    error_size = os.path.getsize(error_log_path) if os.path.exists(error_log_path) else 0
    
    try:
        while True:
            # Check API log
            if os.path.exists(api_log_path):
                current_api_size = os.path.getsize(api_log_path)
                if current_api_size > api_size:
                    # New content added
                    with open(api_log_path, 'r') as f:
                        f.seek(api_size)
                        new_content = f.read()
                        if new_content.strip():
                            lines = new_content.strip().split('\n')
                            for line in lines:
                                if line.strip():
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    print(f"[{timestamp}] 📝 {line.strip()}")
                    api_size = current_api_size
            
            # Check error log
            if os.path.exists(error_log_path):
                current_error_size = os.path.getsize(error_log_path)
                if current_error_size > error_size:
                    # New content added
                    with open(error_log_path, 'r') as f:
                        f.seek(error_size)
                        new_content = f.read()
                        if new_content.strip():
                            lines = new_content.strip().split('\n')
                            for line in lines:
                                if line.strip():
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    print(f"[{timestamp}] ❌ {line.strip()}")
                    error_size = current_error_size
            
            time.sleep(0.1)  # Check every 100ms
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")
        print("\n💡 To see all logs:")
        print("   tail -f logs/api.log")
        print("   tail -f logs/errors.log")

def show_recent_logs(lines=20):
    """Show recent log entries"""
    print(f"📋 Recent Log Entries (last {lines} lines)")
    print("=" * 50)
    
    api_log_path = "logs/api.log"
    if not os.path.exists(api_log_path):
        print("❌ No log file found")
        return
    
    try:
        with open(api_log_path, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            if recent_lines:
                for line in recent_lines:
                    print(line.rstrip())
            else:
                print("📭 No log entries found")
    except Exception as e:
        print(f"❌ Error reading log file: {e}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "recent":
            show_recent_logs()
        elif sys.argv[1] == "monitor":
            monitor_logs()
        else:
            print("Usage:")
            print("  python monitor_logs.py          # Start monitoring")
            print("  python monitor_logs.py recent   # Show recent logs")
            print("  python monitor_logs.py monitor  # Start monitoring")
    else:
        monitor_logs()

if __name__ == "__main__":
    main() 