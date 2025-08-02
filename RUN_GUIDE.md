# 🚀 InterpretabilityWorkbench - How to Run

This guide shows you how to run the complete InterpretabilityWorkbench system with both backend API and frontend UI.

## 📋 Prerequisites

- **Python 3.9+** (we recommend 3.10+)
- **Node.js 18+** and npm
- **Git** (for cloning)

## 🏗️ Project Structure

After reorganization, your project should look like this:
```
InterpretabilityWorkbench/
├── interpretability_workbench/           # Main Python package
│   ├── __init__.py
│   ├── cli.py                           # Command-line interface
│   ├── eval.py                          # SAE evaluation
│   ├── lora_patch.py                    # LoRA patching
│   ├── sae_train.py                     # SAE training
│   ├── trace.py                         # Activation recording
│   └── server/                          # Backend API
│       ├── __init__.py
│       ├── api.py                       # FastAPI server
│       └── websockets.py                # WebSocket handlers
├── ui/                                  # React frontend
│   ├── src/
│   ├── package.json
│   └── ...
├── tests/                               # Python tests
├── pyproject.toml                       # Python package config
├── run.py                              # Simple server starter
└── RUN_GUIDE.md                        # This file
```

## ⚡ Quick Start

### 1. Install Python Dependencies

First, install the package and its dependencies:

```bash
# Install the package in development mode
pip install -e .

# Or if you prefer, install requirements manually:
# pip install fastapi uvicorn torch transformers datasets pyarrow pandas numpy
```

### 2. Start the Backend Server

You have several options to start the backend:

**Option A: Using the simple run script**
```bash
python run.py
```

**Option B: Using the installed command (if working)**
```bash
iwb-server
```

**Option C: Using Python module directly**
```bash
python -m interpretability_workbench.server.api
```

**Option D: Using uvicorn directly**
```bash
uvicorn interpretability_workbench.server.api:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on **http://localhost:8000**

### 3. Build and Start the Frontend

In a **new terminal window**:

```bash
# Navigate to the UI directory
cd ui

# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

The frontend will start on **http://localhost:5173** (or the next available port)

### 4. Build Production Frontend (Optional)

To build the frontend for production and serve it from the backend:

```bash
# Build the React app
cd ui && npm run build

# The built files will be in ui/dist/
# The backend will automatically serve them at http://localhost:8000
```

## 🔧 Development Workflow

### Backend Development
- API server runs with auto-reload enabled
- Make changes to `interpretability_workbench/server/api.py`
- Server automatically restarts on changes
- API docs available at **http://localhost:8000/docs**

### Frontend Development  
- React dev server has hot reload
- Make changes to `ui/src/` files
- Browser automatically updates
- Backend API calls go to **http://localhost:8000**

### Full Stack Development
1. Start backend: `python run.py`
2. Start frontend: `cd ui && npm run dev` 
3. Open **http://localhost:5173** in browser
4. Both servers will auto-reload on changes

## 🎯 Using the Application

### 1. Load a Model
1. Open the settings dialog (⚙️ icon)
2. Enter a HuggingFace model name (e.g., `microsoft/DialoGPT-small`)
3. Click "Load Model" 
4. Wait for model to load (status will show in header)

### 2. Load SAE and Activations
1. After model loads, enter paths to:
   - SAE weights file (`.safetensors` or `.pt`)
   - Activations parquet file (`.parquet`)
2. Click "Load SAE"
3. Features will automatically load

### 3. Browse Features
- Use search box to find specific features
- Filter by layer using dropdown
- Click on features to see detailed analysis
- View top tokens, token clouds, and context examples

### 4. Create Patches
- Click the "+" button next to feature or "Create Patch" in detail view
- Patches appear in the right panel
- Toggle patches on/off with the switch
- Adjust strength with sliders (-5.0 to +5.0)

### 5. Test Inference
- Enter text in the "Inference Tester" 
- Click "Run Inference" to see model predictions
- View probability changes with active patches
- WebSocket provides real-time results (<400ms)

### 6. Export Work
- Export individual features from detail view
- Export all patches from patch panel (coming soon)
- Export trained SAE weights (coming soon)

## 🐛 Troubleshooting

### Backend Issues

**Import Errors**
```bash
# Make sure package is installed
pip install -e .

# Check if modules are importable
python -c "from interpretability_workbench.server.api import app; print('✅ Backend OK')"
```

**Port Already in Use**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process or use different port
uvicorn interpretability_workbench.server.api:app --port 8001
```

**Missing Dependencies**
```bash
# Install all required packages
pip install fastapi uvicorn torch transformers datasets pyarrow pandas numpy python-multipart
```

### Frontend Issues

**Node.js/npm Issues**
```bash
# Check versions
node --version  # Should be 18+
npm --version

# Clear cache and reinstall
cd ui
rm -rf node_modules package-lock.json
npm install
```

**Build Failures**
```bash
# Check for TypeScript errors
cd ui
npm run build

# Fix import paths if needed
npm run lint
```

**API Connection Issues**
- Check that backend is running on port 8000
- Verify CORS is enabled in FastAPI
- Check browser console for network errors
- Ensure environment variables are set correctly

### Common Solutions

**WebSocket Connection Failed**
- Backend server must be running
- Check firewall/proxy settings
- Verify WebSocket endpoint at `ws://localhost:8000/ws`

**Model Loading Timeout** 
- Large models take time to download/load
- Check internet connection for HuggingFace downloads
- Ensure sufficient RAM/VRAM available
- Try smaller models first (e.g., `distilgpt2`)

**File Path Errors**
- Use absolute paths for SAE and activation files
- Ensure files exist and are readable
- Check file formats (`.safetensors`, `.parquet`)

## 📊 Testing

Run the Python tests:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_trace.py

# Run with coverage
pytest --cov=interpretability_workbench
```

## 🔒 Security Notes

- Backend runs on `0.0.0.0:8000` (accessible from network)
- For production, use proper authentication
- CORS is enabled for development
- No secrets should be committed to git

## 🤝 Development Tips

- Use separate terminal windows for backend/frontend
- Backend API docs: **http://localhost:8000/docs**
- Frontend dev tools: **F12** in browser
- Check logs in both terminal windows
- Use `--reload` flags for auto-restart during development

---

**Ready to explore mechanistic interpretability! 🧠✨**

For issues, check the troubleshooting section above or create an issue in the repository.