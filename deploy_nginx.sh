#!/bin/bash

# InterpretabilityWorkbench Nginx Deployment Script
# Run this script from the Interpretabilityworkbench/ directory

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="interpretability-workbench"
NGINX_SITE_NAME="interpretability-workbench"
SERVICE_NAME="interpretability-workbench"
BACKEND_PORT=8000

# Get the current directory (should be Interpretabilityworkbench/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BLUE}🚀 InterpretabilityWorkbench Nginx Deployment Script${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to print colored output
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

# Check if running as root for nginx operations
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root - this is fine for nginx operations"
    else
        print_warning "Some operations may require sudo privileges"
    fi
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if nginx is installed
    if ! command -v nginx &> /dev/null; then
        print_error "Nginx is not installed. Please install nginx first:"
        echo "  Ubuntu/Debian: sudo apt-get install nginx"
        echo "  CentOS/RHEL: sudo yum install nginx"
        exit 1
    fi
    
    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js first."
        exit 1
    fi
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm first."
        exit 1
    fi
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3 first."
        exit 1
    fi
    
    print_status "All prerequisites are installed"
}

# Build the frontend
build_frontend() {
    print_info "Building frontend..."
    
    if [[ ! -d "$PROJECT_ROOT/ui" ]]; then
        print_error "UI directory not found at $PROJECT_ROOT/ui"
        exit 1
    fi
    
    cd "$PROJECT_ROOT/ui"
    
    # Install dependencies if node_modules doesn't exist
    if [[ ! -d "node_modules" ]]; then
        print_info "Installing npm dependencies..."
        npm install
    fi
    
    # Build the project
    print_info "Building React app..."
    npx vite build
    
    if [[ ! -d "dist" ]]; then
        print_error "Build failed - dist directory not created"
        exit 1
    fi
    
    print_status "Frontend built successfully"
    cd "$PROJECT_ROOT"
}

# Create nginx configuration
create_nginx_config() {
    print_info "Creating nginx configuration..."
    
    # Get the absolute path to the dist directory
    DIST_PATH="$PROJECT_ROOT/ui/dist"
    
    # Create nginx config with proper paths
    cat > "$PROJECT_ROOT/nginx.conf" << EOF
server {
    listen 80;
    server_name localhost 34.61.238.238;  # Change this to your domain in production

    # Proxy API requests to FastAPI backend
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Proxy WebSocket connections
    location /ws {
        proxy_pass http://localhost:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Proxy other API endpoints
    location ~ ^/(model|sae|features|patch|inference|export|health|ping|load-model|load-sae) {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Serve static files (React app)
    location / {
        root $DIST_PATH;
        try_files \$uri \$uri/ /index.html;
        index index.html;
        
        # Add cache-busting headers for JavaScript files
        location ~* \.(js|css)$ {
            add_header Cache-Control "no-cache, no-store, must-revalidate";
            add_header Pragma "no-cache";
            add_header Expires "0";
        }
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
} 
EOF
    
    print_status "Nginx configuration created"
}

# Install nginx configuration
install_nginx_config() {
    print_info "Installing nginx configuration..."
    
    # Check if running as root or with sudo
    if [[ $EUID -ne 0 ]]; then
        print_warning "Installing nginx config requires sudo privileges"
        sudo cp "$PROJECT_ROOT/nginx.conf" "/etc/nginx/sites-available/$NGINX_SITE_NAME"
        sudo ln -sf "/etc/nginx/sites-available/$NGINX_SITE_NAME" "/etc/nginx/sites-enabled/"
    else
        cp "$PROJECT_ROOT/nginx.conf" "/etc/nginx/sites-available/$NGINX_SITE_NAME"
        ln -sf "/etc/nginx/sites-available/$NGINX_SITE_NAME" "/etc/nginx/sites-enabled/"
    fi
    
    # Test nginx configuration
    if nginx -t; then
        print_status "Nginx configuration is valid"
    else
        print_error "Nginx configuration is invalid"
        exit 1
    fi
    
    # Reload nginx
    if systemctl reload nginx; then
        print_status "Nginx reloaded successfully"
    else
        print_error "Failed to reload nginx"
        exit 1
    fi
}

# Create systemd service
create_systemd_service() {
    print_info "Creating systemd service..."
    
    # Use the virtual environment Python if it exists, otherwise use system Python
    if [[ -f "$PROJECT_ROOT/.venv/bin/python" ]]; then
        PYTHON_PATH="$PROJECT_ROOT/.venv/bin/python"
        print_info "Using virtual environment Python: $PYTHON_PATH"
    else
        PYTHON_PATH=$(which python3)
        print_warning "Using system Python: $PYTHON_PATH"
    fi
    
    # Create service file
    cat > "$PROJECT_ROOT/$SERVICE_NAME.service" << EOF
[Unit]
Description=InterpretabilityWorkbench FastAPI Backend
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=$PROJECT_ROOT/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$PYTHON_PATH $PROJECT_ROOT/run.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    print_status "Systemd service file created"
}

# Install systemd service
install_systemd_service() {
    print_info "Installing systemd service..."
    
    # Check if running as root or with sudo
    if [[ $EUID -ne 0 ]]; then
        print_warning "Installing systemd service requires sudo privileges"
        sudo cp "$PROJECT_ROOT/$SERVICE_NAME.service" "/etc/systemd/system/"
        sudo systemctl daemon-reload
        sudo systemctl enable "$SERVICE_NAME"
    else
        cp "$PROJECT_ROOT/$SERVICE_NAME.service" "/etc/systemd/system/"
        systemctl daemon-reload
        systemctl enable "$SERVICE_NAME"
    fi
    
    print_status "Systemd service installed and enabled"
}

# Start the service
start_service() {
    print_info "Starting the service..."
    
    if [[ $EUID -ne 0 ]]; then
        sudo systemctl start "$SERVICE_NAME"
    else
        systemctl start "$SERVICE_NAME"
    fi
    
    # Wait a moment for the service to start
    sleep 2
    
    # Check if service is running
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        print_status "Service started successfully"
    else
        print_error "Failed to start service"
        print_info "Check service status with: sudo systemctl status $SERVICE_NAME"
        exit 1
    fi
}

# Test the deployment
test_deployment() {
    print_info "Testing deployment..."
    
    # Wait for service to be ready
    sleep 3
    
    # Test health endpoint
    if curl -s http://localhost:$BACKEND_PORT/health > /dev/null; then
        print_status "Backend health check passed"
    else
        print_warning "Backend health check failed - service may still be starting"
    fi
    
    # Test nginx proxy
    if curl -s http://localhost/health > /dev/null; then
        print_status "Nginx proxy test passed"
    else
        print_warning "Nginx proxy test failed - check nginx configuration"
    fi
    
    print_info "Deployment test completed"
}

# Show status
show_status() {
    print_info "Deployment Status:"
    echo ""
    
    # Service status
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${GREEN}✓ Backend service: RUNNING${NC}"
    else
        echo -e "${RED}✗ Backend service: NOT RUNNING${NC}"
    fi
    
    # Nginx status
    if systemctl is-active --quiet nginx; then
        echo -e "${GREEN}✓ Nginx: RUNNING${NC}"
    else
        echo -e "${RED}✗ Nginx: NOT RUNNING${NC}"
    fi
    
    # Port status
    if netstat -tuln | grep -q ":$BACKEND_PORT "; then
        echo -e "${GREEN}✓ Backend port $BACKEND_PORT: LISTENING${NC}"
    else
        echo -e "${RED}✗ Backend port $BACKEND_PORT: NOT LISTENING${NC}"
    fi
    
    echo ""
    print_info "Useful commands:"
    echo "  View logs: sudo journalctl -u $SERVICE_NAME -f"
    echo "  Restart service: sudo systemctl restart $SERVICE_NAME"
    echo "  Check nginx logs: sudo tail -f /var/log/nginx/error.log"
    echo "  Access app: http://localhost"
}

# Main deployment function
main() {
    echo "Starting deployment from: $PROJECT_ROOT"
    echo ""
    
    check_root
    check_prerequisites
    build_frontend
    create_nginx_config
    install_nginx_config
    create_systemd_service
    install_systemd_service
    start_service
    test_deployment
    show_status
    
    echo ""
    print_status "Deployment completed successfully!"
    print_info "Your app should now be accessible at: http://localhost"
}

# Handle command line arguments
case "${1:-}" in
    "status")
        show_status
        ;;
    "restart")
        print_info "Restarting service..."
        if [[ $EUID -ne 0 ]]; then
            sudo systemctl restart "$SERVICE_NAME"
        else
            systemctl restart "$SERVICE_NAME"
        fi
        show_status
        ;;
    "logs")
        print_info "Showing service logs..."
        if [[ $EUID -ne 0 ]]; then
            sudo journalctl -u "$SERVICE_NAME" -f
        else
            journalctl -u "$SERVICE_NAME" -f
        fi
        ;;
    "test")
        test_deployment
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args)  - Full deployment"
        echo "  status     - Show deployment status"
        echo "  restart    - Restart the service"
        echo "  logs       - Show service logs"
        echo "  test       - Test the deployment"
        echo "  help       - Show this help"
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 