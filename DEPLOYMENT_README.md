# InterpretabilityWorkbench Deployment Guide

## Quick Start (One-Command Deployment)

From the `Interpretabilityworkbench/` directory:

```bash
# Make scripts executable
chmod +x deploy_nginx.sh setup_python_env.sh

# Setup Python environment
./setup_python_env.sh

# Deploy with nginx
./deploy_nginx.sh
```

That's it! Your app will be accessible at `http://localhost`.

## Manual Deployment Steps

If you prefer to do it step by step:

### 1. Prerequisites

Install required software:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install nginx nodejs npm python3 python3-venv

# CentOS/RHEL
sudo yum install nginx nodejs npm python3 python3-venv
```

### 2. Setup Python Environment
```bash
./setup_python_env.sh
```

### 3. Build Frontend
```bash
cd ui
npm install
npm run build
cd ..
```

### 4. Deploy with Nginx
```bash
./deploy_nginx.sh
```

## Deployment Script Commands

The `deploy_nginx.sh` script supports several commands:

```bash
# Full deployment (default)
./deploy_nginx.sh

# Check deployment status
./deploy_nginx.sh status

# Restart the service
./deploy_nginx.sh restart

# View service logs
./deploy_nginx.sh logs

# Test the deployment
./deploy_nginx.sh test

# Show help
./deploy_nginx.sh help
```

## What the Deployment Script Does

1. **Checks Prerequisites**: Verifies nginx, Node.js, npm, and Python are installed
2. **Builds Frontend**: Installs npm dependencies and builds the React app
3. **Creates Nginx Config**: Generates nginx configuration with proper paths
4. **Installs Nginx Config**: Copies config to nginx sites and enables it
5. **Creates Systemd Service**: Generates service file for the FastAPI backend
6. **Installs Service**: Enables and starts the systemd service
7. **Tests Deployment**: Verifies everything is working

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Browser   │───▶│    Nginx    │───▶│  FastAPI    │
│             │    │             │    │  Backend    │
│             │    │ Static Files│    │ Port 8000   │
│             │    │ API Proxy   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

- **Nginx**: Serves static files (React app) and proxies API requests
- **FastAPI**: Handles API endpoints and WebSocket connections
- **Systemd**: Manages the FastAPI service

## Configuration Files

### Nginx Configuration
- **Location**: `/etc/nginx/sites-available/interpretability-workbench`
- **Static Files**: Served from `ui/dist/`
- **API Proxy**: Routes to `localhost:8000`
- **WebSocket**: Proxied to `/ws` endpoint

### Systemd Service
- **Service Name**: `interpretability-workbench`
- **User**: Current user
- **Working Directory**: Project root
- **Auto-restart**: Enabled

## Troubleshooting

### Service Not Starting
```bash
# Check service status
sudo systemctl status interpretability-workbench

# View logs
sudo journalctl -u interpretability-workbench -f

# Check if port is in use
sudo netstat -tuln | grep :8000
```

### Nginx Issues
```bash
# Test nginx config
sudo nginx -t

# Check nginx logs
sudo tail -f /var/log/nginx/error.log

# Restart nginx
sudo systemctl restart nginx
```

### Frontend Not Loading
```bash
# Check if dist directory exists
ls -la ui/dist/

# Rebuild frontend
cd ui && npm run build && cd ..
```

### WebSocket Connection Issues
```bash
# Test WebSocket endpoint
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Key: test" -H "Sec-WebSocket-Version: 13" http://localhost/ws
```

## Production Considerations

### SSL/HTTPS
Add SSL configuration to nginx:
```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    # ... rest of config
}
```

### Domain Configuration
Update the `server_name` in nginx.conf:
```nginx
server_name your-domain.com;
```

### Firewall
Ensure ports 80 (and 443 for HTTPS) are open:
```bash
sudo ufw allow 80
sudo ufw allow 443
```

### Monitoring
Set up monitoring for the service:
```bash
# Check service health
curl http://localhost/health

# Monitor logs
sudo journalctl -u interpretability-workbench -f
```

## File Structure After Deployment

```
Interpretabilityworkbench/
├── deploy_nginx.sh              # Deployment script
├── setup_python_env.sh          # Python setup script
├── nginx.conf                   # Generated nginx config
├── interpretability-workbench.service  # Generated systemd service
├── ui/
│   ├── dist/                    # Built frontend
│   └── ...
├── interpretability_workbench/  # Backend code
└── ...
```

## Environment Variables

The deployment uses relative URLs by default. If you need to configure specific URLs:

1. Create `.env` file in the project root
2. Set environment variables:
   ```
   VITE_API_URL=http://your-domain.com
   VITE_WS_URL=ws://your-domain.com/ws
   ```

## Updating the Deployment

To update the application:

```bash
# Pull latest changes
git pull

# Rebuild frontend
cd ui && npm run build && cd ..

# Restart service
./deploy_nginx.sh restart
```

## Uninstalling

To remove the deployment:

```bash
# Stop and disable service
sudo systemctl stop interpretability-workbench
sudo systemctl disable interpretability-workbench
sudo rm /etc/systemd/system/interpretability-workbench.service

# Remove nginx config
sudo rm /etc/nginx/sites-enabled/interpretability-workbench
sudo rm /etc/nginx/sites-available/interpretability-workbench
sudo systemctl reload nginx
``` 