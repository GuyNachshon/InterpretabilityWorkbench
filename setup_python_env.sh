#!/bin/bash

# Setup Python Environment for InterpretabilityWorkbench
# Run this script from the Interpretabilityworkbench/ directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BLUE}🐍 Python Environment Setup${NC}"
echo -e "${BLUE}=========================${NC}"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Check if virtualenv is available
if ! python3 -m venv --help &> /dev/null; then
    print_error "venv module is not available. Please install python3-venv."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
    print_info "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/.venv"
    print_status "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

# Upgrade pip
print_info "Upgrading pip..."
"$PROJECT_ROOT/.venv/bin/pip" install --upgrade pip

# Install dependencies
print_info "Installing Python dependencies..."
if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    "$PROJECT_ROOT/.venv/bin/pip" install -e .
else
    print_warning "pyproject.toml not found, installing basic dependencies..."
    "$PROJECT_ROOT/.venv/bin/pip" install fastapi uvicorn torch transformers numpy pandas
fi

print_status "Python environment setup completed!"
echo ""
print_info "To activate the environment manually:"
echo "  source $PROJECT_ROOT/.venv/bin/activate"
echo ""
print_info "To run the application:"
echo "  python run.py" 