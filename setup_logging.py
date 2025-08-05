#!/usr/bin/env python3
"""
Setup script for logging configuration
Configures logging for different environments (development, production, etc.)
"""
import os
import sys
import json
import argparse
from pathlib import Path

def create_logging_config(environment="development", log_level="INFO", max_size_mb=10, backup_count=5):
    """Create logging configuration"""
    
    config = {
        "environment": environment,
        "log_level": log_level,
        "log_dir": "logs",
        "max_size_mb": max_size_mb,
        "backup_count": backup_count,
        "loggers": {
            "api": {"level": log_level},
            "model": {"level": log_level},
            "sae": {"level": log_level},
            "training": {"level": log_level},
            "websocket": {"level": log_level}
        },
        "handlers": {
            "console": {
                "enabled": True,
                "level": log_level,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "file": {
                "enabled": True,
                "level": "DEBUG",
                "filename": "logs/api.log",
                "max_bytes": max_size_mb * 1024 * 1024,
                "backup_count": backup_count,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            },
            "error_file": {
                "enabled": True,
                "level": "ERROR",
                "filename": "logs/errors.log",
                "max_bytes": max_size_mb * 1024 * 1024,
                "backup_count": backup_count,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            }
        }
    }
    
    # Environment-specific configurations
    if environment == "production":
        config["log_level"] = "WARNING"
        config["handlers"]["console"]["level"] = "WARNING"
        config["max_size_mb"] = 50
        config["backup_count"] = 10
    elif environment == "development":
        config["log_level"] = "DEBUG"
        config["handlers"]["console"]["level"] = "INFO"
    elif environment == "testing":
        config["log_level"] = "DEBUG"
        config["handlers"]["console"]["enabled"] = False
        config["handlers"]["file"]["filename"] = "logs/test.log"
    
    return config

def setup_logging_directory(config):
    """Create logging directory and files"""
    log_dir = Path(config["log_dir"])
    log_dir.mkdir(exist_ok=True)
    
    # Create log files if they don't exist
    log_files = [
        config["handlers"]["file"]["filename"],
        config["handlers"]["error_file"]["filename"]
    ]
    
    for log_file in log_files:
        log_path = Path(log_file)
        if not log_path.exists():
            log_path.touch()
            print(f"✅ Created log file: {log_file}")
    
    # Set appropriate permissions
    try:
        os.chmod(log_dir, 0o755)
        for log_file in log_files:
            os.chmod(log_file, 0o644)
        print("✅ Set log file permissions")
    except Exception as e:
        print(f"⚠️  Could not set permissions: {e}")

def create_systemd_service():
    """Create systemd service file for production deployment"""
    service_content = """[Unit]
Description=InterpretabilityWorkbench API Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/InterpretabilityWorkbench
Environment=PYTHONPATH=/path/to/InterpretabilityWorkbench
Environment=RELOAD=false
ExecStart=/usr/bin/python3 run.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_file = "interpretability-workbench.service"
    with open(service_file, 'w') as f:
        f.write(service_content)
    
    print(f"✅ Created systemd service file: {service_file}")
    print("📝 Remember to:")
    print("   - Update WorkingDirectory and PYTHONPATH paths")
    print("   - Copy to /etc/systemd/system/")
    print("   - Run: sudo systemctl daemon-reload")
    print("   - Run: sudo systemctl enable interpretability-workbench")

def create_logrotate_config():
    """Create logrotate configuration"""
    logrotate_content = """# Logrotate configuration for InterpretabilityWorkbench
/path/to/InterpretabilityWorkbench/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload interpretability-workbench
    endscript
}
"""
    
    logrotate_file = "interpretability-workbench.logrotate"
    with open(logrotate_file, 'w') as f:
        f.write(logrotate_content)
    
    print(f"✅ Created logrotate config: {logrotate_file}")
    print("📝 Remember to:")
    print("   - Update the path in the configuration")
    print("   - Copy to /etc/logrotate.d/")
    print("   - Test with: sudo logrotate -d /etc/logrotate.d/interpretability-workbench")

def create_monitoring_script():
    """Create a monitoring script for production"""
    monitoring_script = """#!/bin/bash
# Monitoring script for InterpretabilityWorkbench

LOG_DIR="/path/to/InterpretabilityWorkbench/logs"
HEALTH_CHECK_URL="http://localhost:8000/health"
ALERT_EMAIL="admin@example.com"

# Check if server is running
if ! curl -f -s $HEALTH_CHECK_URL > /dev/null; then
    echo "CRITICAL: InterpretabilityWorkbench server is down!"
    # Add your alert mechanism here (email, Slack, etc.)
    exit 1
fi

# Check log file sizes
for log_file in $LOG_DIR/*.log; do
    if [ -f "$log_file" ]; then
        size=$(stat -c%s "$log_file")
        size_mb=$((size / 1024 / 1024))
        if [ $size_mb -gt 100 ]; then
            echo "WARNING: Log file $log_file is large ($size_mb MB)"
        fi
    fi
done

# Check disk space
disk_usage=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $disk_usage -gt 90 ]; then
    echo "CRITICAL: Disk usage is $disk_usage%"
    exit 1
fi

echo "OK: System is healthy"
"""
    
    with open("monitor.sh", 'w') as f:
        f.write(monitoring_script)
    
    os.chmod("monitor.sh", 0o755)
    print("✅ Created monitoring script: monitor.sh")
    print("📝 Remember to:")
    print("   - Update paths in the script")
    print("   - Set up cron job: */5 * * * * /path/to/monitor.sh")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Setup logging for InterpretabilityWorkbench')
    parser.add_argument('--environment', choices=['development', 'production', 'testing'], 
                       default='development', help='Environment to configure')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Log level')
    parser.add_argument('--max-size', type=int, default=10, help='Max log file size in MB')
    parser.add_argument('--backup-count', type=int, default=5, help='Number of backup files')
    parser.add_argument('--systemd', action='store_true', help='Create systemd service file')
    parser.add_argument('--logrotate', action='store_true', help='Create logrotate config')
    parser.add_argument('--monitoring', action='store_true', help='Create monitoring script')
    parser.add_argument('--output', help='Output config file path')
    
    args = parser.parse_args()
    
    print("🔧 Setting up logging for InterpretabilityWorkbench")
    print(f"📋 Environment: {args.environment}")
    print(f"📊 Log Level: {args.log_level}")
    print(f"💾 Max Size: {args.max_size}MB")
    print(f"📦 Backup Count: {args.backup_count}")
    print()
    
    # Create configuration
    config = create_logging_config(
        environment=args.environment,
        log_level=args.log_level,
        max_size_mb=args.max_size,
        backup_count=args.backup_count
    )
    
    # Setup logging directory
    setup_logging_directory(config)
    
    # Save configuration
    output_file = args.output or f"logging_config_{args.environment}.json"
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved configuration to: {output_file}")
    
    # Create additional files if requested
    if args.systemd:
        create_systemd_service()
    
    if args.logrotate:
        create_logrotate_config()
    
    if args.monitoring:
        create_monitoring_script()
    
    print("\n🎉 Logging setup complete!")
    print("\n📝 Next steps:")
    print("   1. Start the server: python run.py")
    print("   2. Check logs: tail -f logs/api.log")
    print("   3. Run health check: python health_check.py")
    print("   4. Monitor logs: python monitor_logs.py")

if __name__ == "__main__":
    main() 