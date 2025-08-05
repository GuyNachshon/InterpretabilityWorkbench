# Logging and Progress Tracking Features

This document describes the enhanced logging and progress tracking features added to the InterpretabilityWorkbench.

## 🪵 Logging System

### Overview
The application now includes a comprehensive logging system with structured logging for different components:

- **API Logging**: All API endpoints and requests
- **Model Logging**: Model loading and operations
- **SAE Logging**: Sparse Autoencoder operations
- **Training Logging**: SAE training progress and metrics
- **WebSocket Logging**: Real-time communication

### Log Files
Logs are stored in the `logs/` directory:

- `logs/api.log` - All application logs (rotating, max 10MB)
- `logs/errors.log` - Error logs only (rotating, max 10MB)

### Log Format
```
2024-01-15 10:30:45 - api - INFO - Health check requested
2024-01-15 10:30:46 - model - INFO - Loading model openai-community/gpt2...
2024-01-15 10:30:47 - training - INFO - Starting SAE training for layer 6
```

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information about application flow
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failed operations

## 📊 Progress Tracking

### Real-time Training Progress
SAE training now provides real-time progress updates via WebSocket:

#### Progress Information
- **Current Epoch**: Current training epoch
- **Total Epochs**: Total number of epochs
- **Progress Percentage**: 0-100% completion
- **Training Metrics**: Loss values, reconstruction loss, sparsity loss
- **Time Estimates**: Elapsed time and estimated time remaining

#### WebSocket Messages
```json
{
  "type": "training_progress",
  "job_id": "uuid-1234",
  "status": "training",
  "epoch": 5,
  "total_epochs": 100,
  "progress": 5.0,
  "metrics": {
    "train_loss": 0.1234,
    "reconstruction_loss": 0.0987,
    "sparsity_loss": 0.0247
  },
  "elapsed_time": 300.5,
  "estimated_remaining": 5700.0
}
```

### UI Progress Indicators
The UI displays progress cards showing:

- **Progress Bar**: Visual progress indicator
- **Status Badge**: Color-coded status (starting, in_progress, completed, failed)
- **Metrics Display**: Real-time training metrics
- **Time Estimates**: ETA for completion
- **Error Messages**: Detailed error information if training fails

## 🔧 Configuration

### Logging Configuration
Logging is automatically configured when the server starts:

```python
# Logs are stored in logs/ directory
# Console output for INFO level and above
# File output for DEBUG level and above
# Error logs separated for easier debugging
```

### WebSocket Progress Updates
Progress updates are automatically sent during training:

```python
# Training progress callback sends updates every epoch
class TrainingProgressCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Calculate progress and send via WebSocket
        progress = (current_epoch + 1) / total_epochs * 100
        # Broadcast to all connected clients
```

## 🚀 Usage Examples

### Starting the Server with Logging
```bash
# Start the server (logging is automatic)
python run.py

# Check logs in real-time
tail -f logs/api.log

# Monitor errors
tail -f logs/errors.log
```

### Training with Progress Tracking
1. **Start Training**: Use the UI or API to start SAE training
2. **Monitor Progress**: Watch real-time progress in the UI
3. **Check Logs**: Monitor detailed logs for debugging

```python
# API call to start training
response = requests.post("/sae/train", json={
    "layer_idx": 6,
    "activation_data_path": "activations.parquet",
    "latent_dim": 16384,
    "max_epochs": 100
})

# WebSocket will receive progress updates
# UI will show real-time progress
```

### WebSocket Connection
```javascript
// Connect to WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    if (message.type === 'training_progress') {
        console.log(`Training ${message.progress}% complete`);
        console.log(`Current epoch: ${message.epoch}/${message.total_epochs}`);
        console.log(`Loss: ${message.metrics.train_loss}`);
    }
};
```

## 🧪 Testing

Run the test script to verify functionality:

```bash
python test_logging_and_progress.py
```

This will test:
- ✅ Logging system functionality
- ✅ WebSocket communication
- ✅ Progress tracking
- ✅ Error handling

## 📈 Benefits

### For Developers
- **Debugging**: Comprehensive logs for troubleshooting
- **Monitoring**: Real-time progress tracking
- **Error Handling**: Detailed error information
- **Performance**: Track training metrics and timing

### For Users
- **Transparency**: See exactly what's happening during training
- **Progress**: Know how long training will take
- **Confidence**: Real-time feedback on training health
- **Debugging**: Clear error messages when things go wrong

## 🔍 Troubleshooting

### Common Issues

1. **Logs not appearing**
   - Check `logs/` directory exists
   - Verify write permissions
   - Check disk space

2. **Progress not updating**
   - Verify WebSocket connection
   - Check browser console for errors
   - Ensure training job is active

3. **High memory usage**
   - Log rotation should prevent this
   - Check log file sizes
   - Restart server if needed

### Debug Commands
```bash
# Check log file sizes
ls -lh logs/

# Monitor logs in real-time
tail -f logs/api.log

# Search for errors
grep ERROR logs/api.log

# Check WebSocket connections
netstat -an | grep :8000
```

## 📝 Future Enhancements

- **Log Aggregation**: Centralized logging for multiple instances
- **Metrics Dashboard**: Web-based metrics visualization
- **Alert System**: Notifications for training completion/failure
- **Performance Profiling**: Detailed timing analysis
- **Custom Log Levels**: User-configurable logging verbosity 