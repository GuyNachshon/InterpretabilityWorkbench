# Nginx Deployment Guide

## Overview
This guide covers deploying the InterpretabilityWorkbench with nginx serving the frontend and proxying API requests to the FastAPI backend.

## Architecture
- **Nginx**: Serves static files (React app) and proxies API requests
- **FastAPI**: Runs on localhost:8000, handles API endpoints and WebSocket connections
- **React App**: Built static files served by nginx

## Setup Steps

### 1. Build the Frontend
```bash
cd ui
npm install
npm run build
```

### 2. Configure Nginx
Copy the `nginx.conf` file to your nginx sites directory:
```bash
sudo cp nginx.conf /etc/nginx/sites-available/interpretability-workbench
sudo ln -s /etc/nginx/sites-available/interpretability-workbench /etc/nginx/sites-enabled/
```

**Important**: Update the paths in `nginx.conf`:
- Replace `your-domain.com` with your actual domain
- Replace `/path/to/your/app/ui/dist` with the actual path to your built frontend

### 3. Start the FastAPI Backend
```bash
# Run the backend server
python run.py
# Or with uvicorn directly
uvicorn interpretability_workbench.server.api:app --host 0.0.0.0 --port 8000
```

### 4. Test and Reload Nginx
```bash
sudo nginx -t  # Test configuration
sudo systemctl reload nginx  # Reload nginx
```

## Configuration Details

### Nginx Configuration
The nginx config handles:
- **Static Files**: Serves the React app from `/ui/dist`
- **API Proxy**: Routes `/api/*` requests to FastAPI backend
- **WebSocket**: Proxies `/ws` connections with proper headers
- **SPA Routing**: Falls back to `index.html` for client-side routes

### FastAPI Backend
The backend now only handles:
- API endpoints (`/model/*`, `/sae/*`, `/features/*`, etc.)
- WebSocket connections (`/ws`)
- No static file serving (handled by nginx)

## Troubleshooting

### WebSocket Connection Issues
If WebSocket connections fail:
1. Check nginx logs: `sudo tail -f /var/log/nginx/error.log`
2. Verify WebSocket proxy configuration in nginx
3. Ensure FastAPI is running on port 8000

### Static File Issues
If the frontend doesn't load:
1. Verify the path in nginx.conf points to the correct `ui/dist` directory
2. Check file permissions: `ls -la /path/to/ui/dist`
3. Check nginx error logs

### API Proxy Issues
If API calls fail:
1. Verify FastAPI is running: `curl http://localhost:8000/health`
2. Check nginx proxy configuration
3. Verify the proxy_pass URL in nginx.conf

## Production Considerations

### SSL/HTTPS
Add SSL configuration to nginx.conf:
```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    # ... rest of config
}
```

### Process Management
Use a process manager like systemd or supervisor to keep the FastAPI backend running:
```bash
# Create systemd service
sudo nano /etc/systemd/system/interpretability-workbench.service
```

### Logging
Configure proper logging for both nginx and FastAPI:
- Nginx: `/var/log/nginx/`
- FastAPI: Configure logging in the Python app

## Environment Variables
No changes needed to environment variables. The frontend will use relative URLs which work correctly with the nginx proxy setup. 