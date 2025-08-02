#!/usr/bin/env python3
"""
InterpretabilityWorkbench CLI - Entry point for trace|train|ui commands
"""
import argparse
import sys
from pathlib import Path


def trace_command(args):
    """Record activations from model layers"""
    from trace import ActivationRecorder
    
    recorder = ActivationRecorder(
        model_name=args.model,
        layer_idx=args.layer,
        output_path=args.out
    )
    recorder.record()


def train_command(args):
    """Train sparse autoencoder on recorded activations"""
    from sae_train import SAETrainer
    
    trainer = SAETrainer(
        activation_path=args.activations,
        output_dir=args.out,
        layer_idx=args.layer
    )
    trainer.train()


def ui_command(args):
    """Launch web UI server"""
    from server.api import app
    import uvicorn
    
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(description="InterpretabilityWorkbench")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Trace command
    trace_parser = subparsers.add_parser('trace', help='Record model activations')
    trace_parser.add_argument('--model', required=True, help='Model name or path')
    trace_parser.add_argument('--layer', type=int, required=True, help='Layer index to record')
    trace_parser.add_argument('--out', required=True, help='Output parquet file path')
    trace_parser.add_argument('--dataset', default='openwebtext', help='Dataset to use')
    trace_parser.add_argument('--max-samples', type=int, default=10000, help='Maximum samples to process')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train sparse autoencoder')
    train_parser.add_argument('--activations', required=True, help='Input activations parquet file')
    train_parser.add_argument('--out', required=True, help='Output directory for SAE weights')
    train_parser.add_argument('--layer', type=int, required=True, help='Layer index')
    train_parser.add_argument('--latent-dim', type=int, default=16384, help='SAE latent dimension')
    train_parser.add_argument('--sparsity-coef', type=float, default=1e-3, help='Sparsity coefficient (β)')
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Launch web interface')
    ui_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    ui_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    ui_parser.add_argument('--model', help='Model to load for live editing')
    ui_parser.add_argument('--sae-dir', help='Directory containing SAE weights')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'trace':
        trace_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'ui':
        ui_command(args)


if __name__ == '__main__':
    main()