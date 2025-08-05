# Verification Guide: Logging and Progress Tracking

This guide provides multiple ways to verify that the logging and progress tracking features are working correctly **without running the full training process**.

## 🚀 Quick Start Verification

### 1. Quick Check (30 seconds)
```bash
# Run the quick verification script
python quick_verify.py
```

This will check:
- ✅ Server connectivity
- ✅ Logging system setup
- ✅ API endpoint functionality
- ✅ WebSocket communication
- ✅ Training infrastructure readiness

### 2. Comprehensive Testing (2-3 minutes)
```bash
# Run the full test suite
python test_logging_and_progress.py
```

This provides detailed testing of all components with pass/fail results.

## 📊 Real-time Monitoring

### 3. Watch Logs Live
```bash
# Monitor logs in real-time
python monitor_logs.py

# Or show recent logs
python monitor_logs.py recent
```

### 4. Manual Log Inspection
```bash
# Check if logs exist
ls -la logs/

# View recent API logs
tail -20 logs/api.log

# View recent errors
tail -10 logs/errors.log

# Monitor logs in real-time
tail -f logs/api.log
```

## 🔍 Step-by-Step Verification

### Step 1: Start the Server
```bash
# Start the server
python run.py
```

**Expected Output:**
```
INFO - api - Starting InterpretabilityWorkbench API server
INFO - api - Server configuration: host=0.0.0.0, port=8000, reload=False
```

### Step 2: Check Log Files
```bash
# Verify logs directory exists
ls -la logs/

# Should show:
# logs/
# ├── api.log
# └── errors.log
```

### Step 3: Test API Endpoints
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test model status
curl http://localhost:8000/model/status

# Test SAE status
curl http://localhost:8000/sae/status
```

**Expected Log Entries:**
```
2024-01-15 10:30:45 - api - DEBUG - Health check requested
2024-01-15 10:30:46 - api - DEBUG - Model status requested
2024-01-15 10:30:47 - api - DEBUG - SAE status requested
```

### Step 4: Test WebSocket Connection
```bash
# Use the quick verification script
python quick_verify.py
```

**Expected Output:**
```
✅ WebSocket communication working
```

### Step 5: Test Training Infrastructure
```bash
# Test training endpoint (will fail but should log)
curl -X POST http://localhost:8000/sae/train \
  -H "Content-Type: application/json" \
  -d '{
    "layer_idx": 6,
    "activation_data_path": "nonexistent.parquet",
    "latent_dim": 1024,
    "sparsity_coef": 0.001,
    "learning_rate": 0.001,
    "max_epochs": 5,
    "tied_weights": true,
    "activation_fn": "relu"
  }'
```

**Expected Log Entries:**
```
2024-01-15 10:30:48 - training - INFO - Starting SAE training for layer 6
2024-01-15 10:30:49 - training - ERROR - Failed to start SAE training: ...
```

## 🎯 What to Look For

### ✅ Success Indicators

1. **Log Files Exist:**
   ```
   logs/
   ├── api.log      # Contains all application logs
   └── errors.log   # Contains only error logs
   ```

2. **Log Format is Correct:**
   ```
   2024-01-15 10:30:45 - api - INFO - Starting InterpretabilityWorkbench API server
   2024-01-15 10:30:46 - model - INFO - Loading model openai-community/gpt2...
   2024-01-15 10:30:47 - training - INFO - Starting SAE training for layer 6
   ```

3. **WebSocket Communication:**
   - Connection established
   - Ping/pong messages working
   - No connection errors

4. **API Endpoints Respond:**
   - Health check returns 200
   - Model status returns 200
   - SAE status returns 200

5. **Training Infrastructure:**
   - Training endpoint accepts requests
   - Proper error handling for missing files
   - Logs training attempts

### ❌ Failure Indicators

1. **Missing Log Files:**
   ```
   ❌ Logs directory missing
   ❌ API log file missing
   ❌ Error log file missing
   ```

2. **Server Not Running:**
   ```
   ❌ Cannot connect to server: Connection refused
   ```

3. **WebSocket Issues:**
   ```
   ❌ WebSocket connection failed
   ❌ No WebSocket messages received
   ```

4. **Logging Not Working:**
   ```
   ❌ No new log entries from API calls
   ❌ Log format structure incorrect
   ```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Server Won't Start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process
pkill -f "python run.py"

# Start server again
python run.py
```

#### 2. Logs Not Appearing
```bash
# Check file permissions
ls -la logs/

# Fix permissions if needed
chmod 755 logs/
chmod 644 logs/*.log

# Check disk space
df -h .
```

#### 3. WebSocket Connection Fails
```bash
# Check if WebSocket endpoint is accessible
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Version: 13" \
     -H "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==" \
     http://localhost:8000/ws
```

#### 4. API Endpoints Not Responding
```bash
# Check server logs
tail -f logs/api.log

# Test with curl
curl -v http://localhost:8000/health
```

## 📈 Progress Tracking Verification

### UI Progress Indicators
1. **Start the UI:**
   ```bash
   cd ui
   npm run dev
   ```

2. **Check Progress Components:**
   - Progress cards appear in top-right corner
   - Status badges show correct colors
   - Progress bars update in real-time

3. **Test Training Progress:**
   - Start a training job via UI
   - Watch progress cards update
   - Check WebSocket messages in browser console

### WebSocket Progress Messages
Monitor WebSocket messages in browser console:
```javascript
// In browser console
ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'training_progress') {
        console.log('Progress:', message.progress + '%');
        console.log('Epoch:', message.epoch + '/' + message.total_epochs);
        console.log('Loss:', message.metrics.train_loss);
    }
};
```

## 🎉 Success Checklist

- [ ] Server starts without errors
- [ ] Log files are created in `logs/` directory
- [ ] API endpoints respond correctly
- [ ] WebSocket connection established
- [ ] Log entries appear for API calls
- [ ] Training endpoint accepts requests
- [ ] Error logging works for failed requests
- [ ] UI progress indicators display correctly
- [ ] WebSocket progress messages received

## 📝 Next Steps

Once verification is complete:

1. **Start Training:** Use the UI or API to begin actual SAE training
2. **Monitor Progress:** Watch real-time progress updates
3. **Check Logs:** Monitor detailed logs for debugging
4. **Verify Metrics:** Confirm training metrics are accurate

## 🆘 Getting Help

If verification fails:

1. **Check the logs:** `tail -f logs/api.log`
2. **Run comprehensive tests:** `python test_logging_and_progress.py`
3. **Review error messages:** `tail -f logs/errors.log`
4. **Check system resources:** `df -h . && free -h`

The logging and progress tracking system is designed to be robust and provide clear feedback about what's working and what isn't. 