#!/usr/bin/env python3
"""
Health check script for InterpretabilityWorkbench
Can be used for monitoring, alerting, and system status checks
"""
import requests
import json
import time
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

class HealthChecker:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
        
    def check_server_health(self):
        """Check if server is responding"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            self.results['server_health'] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.results['server_health'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_model_status(self):
        """Check model loading status"""
        try:
            response = requests.get(f"{self.base_url}/model/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.results['model_status'] = {
                    'status': data.get('status', 'unknown'),
                    'model_name': data.get('model_name'),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                self.results['model_status'] = {
                    'status': 'error',
                    'status_code': response.status_code,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            self.results['model_status'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_sae_status(self):
        """Check SAE status"""
        try:
            response = requests.get(f"{self.base_url}/sae/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.results['sae_status'] = {
                    'status': data.get('status', 'unknown'),
                    'layer_count': data.get('layerCount', 0),
                    'feature_count': data.get('featureCount', 0),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                self.results['sae_status'] = {
                    'status': 'error',
                    'status_code': response.status_code,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            self.results['sae_status'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_log_files(self):
        """Check log file health"""
        log_dir = Path("logs")
        self.results['log_files'] = {
            'logs_directory_exists': log_dir.exists(),
            'api_log_exists': (log_dir / "api.log").exists(),
            'error_log_exists': (log_dir / "errors.log").exists(),
            'timestamp': datetime.now().isoformat()
        }
        
        if (log_dir / "api.log").exists():
            try:
                stat = (log_dir / "api.log").stat()
                self.results['log_files']['api_log_size'] = stat.st_size
                self.results['log_files']['api_log_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                
                # Check if log is being written to recently (within last 5 minutes)
                modified_time = datetime.fromtimestamp(stat.st_mtime)
                if datetime.now() - modified_time < timedelta(minutes=5):
                    self.results['log_files']['api_log_recent'] = True
                else:
                    self.results['log_files']['api_log_recent'] = False
            except Exception as e:
                self.results['log_files']['api_log_error'] = str(e)
    
    def check_training_jobs(self):
        """Check active training jobs"""
        try:
            response = requests.get(f"{self.base_url}/sae/training/jobs", timeout=5)
            if response.status_code == 200:
                data = response.json()
                active_jobs = [job for job in data.get('jobs', []) if job.get('status') in ['starting', 'training']]
                self.results['training_jobs'] = {
                    'total_jobs': len(data.get('jobs', [])),
                    'active_jobs': len(active_jobs),
                    'jobs': data.get('jobs', []),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                self.results['training_jobs'] = {
                    'status': 'error',
                    'status_code': response.status_code,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            self.results['training_jobs'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_disk_space(self):
        """Check disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            self.results['disk_space'] = {
                'total_gb': total / (1024**3),
                'used_gb': used / (1024**3),
                'free_gb': free / (1024**3),
                'usage_percent': (used / total) * 100,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.results['disk_space'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_checks(self):
        """Run all health checks"""
        print("🔍 Running health checks...")
        
        self.check_server_health()
        self.check_model_status()
        self.check_sae_status()
        self.check_log_files()
        self.check_training_jobs()
        self.check_disk_space()
        
        return self.results
    
    def get_summary(self):
        """Get a summary of health status"""
        summary = {
            'overall_status': 'healthy',
            'checks_passed': 0,
            'checks_failed': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check server health
        if self.results.get('server_health', {}).get('status') == 'healthy':
            summary['checks_passed'] += 1
        else:
            summary['checks_failed'] += 1
            summary['overall_status'] = 'unhealthy'
        
        # Check log files
        log_files = self.results.get('log_files', {})
        if log_files.get('logs_directory_exists') and log_files.get('api_log_exists'):
            summary['checks_passed'] += 1
        else:
            summary['checks_failed'] += 1
            summary['overall_status'] = 'unhealthy'
        
        # Check disk space
        disk_space = self.results.get('disk_space', {})
        if 'usage_percent' in disk_space and disk_space['usage_percent'] < 90:
            summary['checks_passed'] += 1
        else:
            summary['checks_failed'] += 1
            if disk_space.get('usage_percent', 0) >= 90:
                summary['overall_status'] = 'warning'
        
        return summary
    
    def print_results(self, format='text'):
        """Print results in specified format"""
        if format == 'json':
            print(json.dumps(self.results, indent=2))
        else:
            self._print_text_results()
    
    def _print_text_results(self):
        """Print results in human-readable text format"""
        print("\n" + "="*60)
        print("🏥 InterpretabilityWorkbench Health Check")
        print("="*60)
        
        # Server Health
        server_health = self.results.get('server_health', {})
        status = server_health.get('status', 'unknown')
        if status == 'healthy':
            print(f"✅ Server Health: {status}")
            print(f"   Response Time: {server_health.get('response_time', 0):.3f}s")
        else:
            print(f"❌ Server Health: {status}")
            if 'error' in server_health:
                print(f"   Error: {server_health['error']}")
        
        # Model Status
        model_status = self.results.get('model_status', {})
        model_state = model_status.get('status', 'unknown')
        print(f"🤖 Model Status: {model_state}")
        if model_status.get('model_name'):
            print(f"   Model: {model_status['model_name']}")
        
        # SAE Status
        sae_status = self.results.get('sae_status', {})
        sae_state = sae_status.get('status', 'unknown')
        print(f"🧠 SAE Status: {sae_state}")
        if sae_state == 'ready':
            print(f"   Layers: {sae_status.get('layer_count', 0)}")
            print(f"   Features: {sae_status.get('feature_count', 0)}")
        
        # Log Files
        log_files = self.results.get('log_files', {})
        if log_files.get('logs_directory_exists'):
            print("✅ Log Directory: Exists")
        else:
            print("❌ Log Directory: Missing")
        
        if log_files.get('api_log_exists'):
            print("✅ API Log: Exists")
            if 'api_log_size' in log_files:
                size_mb = log_files['api_log_size'] / (1024 * 1024)
                print(f"   Size: {size_mb:.2f} MB")
            if log_files.get('api_log_recent'):
                print("   Status: Recently updated")
            else:
                print("   Status: Not recently updated")
        else:
            print("❌ API Log: Missing")
        
        # Training Jobs
        training_jobs = self.results.get('training_jobs', {})
        if 'active_jobs' in training_jobs:
            print(f"🔄 Training Jobs: {training_jobs['active_jobs']} active, {training_jobs['total_jobs']} total")
        else:
            print("❌ Training Jobs: Unable to check")
        
        # Disk Space
        disk_space = self.results.get('disk_space', {})
        if 'usage_percent' in disk_space:
            usage = disk_space['usage_percent']
            free_gb = disk_space['free_gb']
            print(f"💾 Disk Space: {usage:.1f}% used, {free_gb:.1f} GB free")
            if usage > 90:
                print("   ⚠️  Warning: Disk space low!")
        else:
            print("❌ Disk Space: Unable to check")
        
        # Summary
        summary = self.get_summary()
        print("\n" + "-"*60)
        print(f"📊 Summary: {summary['overall_status'].upper()}")
        print(f"   Passed: {summary['checks_passed']}")
        print(f"   Failed: {summary['checks_failed']}")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Health check for InterpretabilityWorkbench')
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL for the API')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    parser.add_argument('--save', help='Save results to file')
    
    args = parser.parse_args()
    
    checker = HealthChecker(args.url)
    results = checker.run_all_checks()
    
    if args.format == 'json':
        checker.print_results('json')
    else:
        checker.print_results('text')
    
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.save}")
    
    # Exit with appropriate code
    summary = checker.get_summary()
    if summary['overall_status'] == 'healthy':
        sys.exit(0)
    elif summary['overall_status'] == 'warning':
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == "__main__":
    main() 