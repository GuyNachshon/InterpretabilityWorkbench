#!/usr/bin/env python3
"""
Log analysis script for InterpretabilityWorkbench
Analyzes logs to identify patterns, errors, and performance issues
"""
import re
import json
import argparse
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

class LogAnalyzer:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.api_log = self.log_dir / "api.log"
        self.error_log = self.log_dir / "errors.log"
        
    def parse_log_line(self, line):
        """Parse a single log line"""
        # Expected format: timestamp - logger - level - message
        pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (\w+) - (\w+) - (.+)$'
        match = re.match(pattern, line.strip())
        
        if match:
            timestamp_str, logger, level, message = match.groups()
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                return {
                    'timestamp': timestamp,
                    'logger': logger,
                    'level': level,
                    'message': message,
                    'raw': line.strip()
                }
            except ValueError:
                return None
        return None
    
    def load_logs(self, hours=None):
        """Load logs from files"""
        entries = []
        
        # Load API logs
        if self.api_log.exists():
            with open(self.api_log, 'r') as f:
                for line in f:
                    entry = self.parse_log_line(line)
                    if entry:
                        entries.append(entry)
        
        # Load error logs
        if self.error_log.exists():
            with open(self.error_log, 'r') as f:
                for line in f:
                    entry = self.parse_log_line(line)
                    if entry:
                        entries.append(entry)
        
        # Filter by time if specified
        if hours:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            entries = [entry for entry in entries if entry['timestamp'] > cutoff_time]
        
        return sorted(entries, key=lambda x: x['timestamp'])
    
    def analyze_logs(self, hours=None):
        """Analyze logs and return statistics"""
        entries = self.load_logs(hours)
        
        if not entries:
            return {"error": "No log entries found"}
        
        # Basic statistics
        stats = {
            'total_entries': len(entries),
            'time_range': {
                'start': entries[0]['timestamp'].isoformat(),
                'end': entries[-1]['timestamp'].isoformat()
            },
            'loggers': Counter(entry['logger'] for entry in entries),
            'levels': Counter(entry['level'] for entry in entries),
            'errors': [],
            'warnings': [],
            'performance_issues': [],
            'patterns': defaultdict(int)
        }
        
        # Analyze each entry
        for entry in entries:
            # Collect errors
            if entry['level'] == 'ERROR':
                stats['errors'].append({
                    'timestamp': entry['timestamp'].isoformat(),
                    'logger': entry['logger'],
                    'message': entry['message']
                })
            
            # Collect warnings
            if entry['level'] == 'WARNING':
                stats['warnings'].append({
                    'timestamp': entry['timestamp'].isoformat(),
                    'logger': entry['logger'],
                    'message': entry['message']
                })
            
            # Look for performance patterns
            if 'timeout' in entry['message'].lower():
                stats['performance_issues'].append({
                    'type': 'timeout',
                    'timestamp': entry['timestamp'].isoformat(),
                    'message': entry['message']
                })
            
            if 'slow' in entry['message'].lower() or 'latency' in entry['message'].lower():
                stats['performance_issues'].append({
                    'type': 'slow_response',
                    'timestamp': entry['timestamp'].isoformat(),
                    'message': entry['message']
                })
            
            # Count message patterns
            message_key = self.extract_message_pattern(entry['message'])
            stats['patterns'][message_key] += 1
        
        return stats
    
    def extract_message_pattern(self, message):
        """Extract a pattern from a message for grouping"""
        # Remove timestamps and IDs to group similar messages
        pattern = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '[TIMESTAMP]', message)
        pattern = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '[UUID]', pattern)
        pattern = re.sub(r'\d+\.\d+', '[NUMBER]', pattern)
        return pattern
    
    def generate_report(self, hours=None, output_file=None):
        """Generate a comprehensive log analysis report"""
        stats = self.analyze_logs(hours)
        
        if 'error' in stats:
            print(f"❌ {stats['error']}")
            return
        
        report = []
        report.append("=" * 80)
        report.append("📊 LOG ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📈 Time Range: {stats['time_range']['start']} to {stats['time_range']['end']}")
        report.append(f"📝 Total Entries: {stats['total_entries']}")
        report.append("")
        
        # Logger distribution
        report.append("🔍 LOGGER DISTRIBUTION:")
        for logger, count in stats['loggers'].most_common():
            percentage = (count / stats['total_entries']) * 100
            report.append(f"   {logger}: {count} entries ({percentage:.1f}%)")
        report.append("")
        
        # Log level distribution
        report.append("📊 LOG LEVEL DISTRIBUTION:")
        for level, count in stats['levels'].most_common():
            percentage = (count / stats['total_entries']) * 100
            report.append(f"   {level}: {count} entries ({percentage:.1f}%)")
        report.append("")
        
        # Errors
        if stats['errors']:
            report.append("❌ ERRORS FOUND:")
            for error in stats['errors'][:10]:  # Show first 10 errors
                report.append(f"   [{error['timestamp']}] {error['logger']}: {error['message']}")
            if len(stats['errors']) > 10:
                report.append(f"   ... and {len(stats['errors']) - 10} more errors")
            report.append("")
        
        # Warnings
        if stats['warnings']:
            report.append("⚠️  WARNINGS FOUND:")
            for warning in stats['warnings'][:10]:  # Show first 10 warnings
                report.append(f"   [{warning['timestamp']}] {warning['logger']}: {warning['message']}")
            if len(stats['warnings']) > 10:
                report.append(f"   ... and {len(stats['warnings']) - 10} more warnings")
            report.append("")
        
        # Performance issues
        if stats['performance_issues']:
            report.append("🐌 PERFORMANCE ISSUES:")
            for issue in stats['performance_issues']:
                report.append(f"   [{issue['timestamp']}] {issue['type']}: {issue['message']}")
            report.append("")
        
        # Common patterns
        report.append("🔄 COMMON MESSAGE PATTERNS:")
        for pattern, count in sorted(stats['patterns'].items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / stats['total_entries']) * 100
            report.append(f"   {count}x ({percentage:.1f}%): {pattern}")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        if stats['errors']:
            report.append("   • Review error logs for recurring issues")
        if stats['warnings']:
            report.append("   • Address warnings to prevent future errors")
        if stats['performance_issues']:
            report.append("   • Investigate performance issues and optimize")
        if stats['levels']['ERROR'] > stats['levels']['INFO']:
            report.append("   • High error rate detected - review system health")
        if not stats['errors'] and not stats['warnings']:
            report.append("   • System appears healthy - no issues detected")
        report.append("")
        
        report.append("=" * 80)
        
        # Print or save report
        report_text = "\n".join(report)
        print(report_text)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"💾 Report saved to: {output_file}")
        
        return stats
    
    def create_visualizations(self, hours=None, output_dir="log_analysis"):
        """Create visualizations of log data"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            print("❌ matplotlib and pandas required for visualizations")
            return
        
        entries = self.load_logs(hours)
        if not entries:
            print("❌ No log entries found for visualization")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(entries)
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.date
        
        # 1. Log level distribution pie chart
        plt.figure(figsize=(10, 6))
        level_counts = df['level'].value_counts()
        plt.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%')
        plt.title('Log Level Distribution')
        plt.savefig(output_path / 'log_levels.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Logger distribution bar chart
        plt.figure(figsize=(10, 6))
        logger_counts = df['logger'].value_counts()
        plt.bar(logger_counts.index, logger_counts.values)
        plt.title('Logger Distribution')
        plt.xlabel('Logger')
        plt.ylabel('Number of Entries')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'logger_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Timeline of log entries
        plt.figure(figsize=(12, 6))
        hourly_counts = df.groupby('hour').size()
        plt.plot(hourly_counts.index, hourly_counts.values, marker='o')
        plt.title('Log Entries by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Entries')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Error timeline
        error_df = df[df['level'] == 'ERROR']
        if not error_df.empty:
            plt.figure(figsize=(12, 6))
            error_hourly = error_df.groupby('hour').size()
            plt.plot(error_hourly.index, error_hourly.values, marker='o', color='red')
            plt.title('Errors by Hour')
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Errors')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'error_timeline.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Visualizations saved to: {output_path}")
        print(f"   • log_levels.png - Log level distribution")
        print(f"   • logger_distribution.png - Logger usage")
        print(f"   • timeline.png - Log entries over time")
        if not error_df.empty:
            print(f"   • error_timeline.png - Error frequency")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze InterpretabilityWorkbench logs')
    parser.add_argument('--hours', type=int, help='Analyze logs from last N hours')
    parser.add_argument('--output', help='Save report to file')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--log-dir', default='logs', help='Log directory path')
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(args.log_dir)
    
    # Generate report
    stats = analyzer.generate_report(hours=args.hours, output_file=args.output)
    
    # Create visualizations if requested
    if args.visualize:
        analyzer.create_visualizations(hours=args.hours)

if __name__ == "__main__":
    main() 