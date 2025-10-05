# 🎮 Gesture-Controlled Driving Game 

## ✅ Current Status

### Dataset: ✅ READY
- **Status**: Fully generated and validated
- **Training Samples**: 120,000+
- **Validation Samples**: 15,000+
- **Test Samples**: 15,000+
- **Classes**: 5 (steer_left, steer_right, accelerate, brake, none)
- **Samples per Class**: 30,000 (balanced)
- **Location**: `data/` directory

### Model: ⏳ NEEDS TRAINING
- **Status**: Not yet trained
- **Next Step**: Train the model using control panel
- **Expected Time**: 10-20 minutes (depends on hardware)
- **Expected Accuracy**: 95%+

### Frontend: ✅ READY
- **Control Panel**: Fully functional web interface
- **Game Interface**: Complete 3D racing game
- **Integration**: All components connected

## 🚀 How to Run the Complete System

### Step 1: Install Dependencies (One-time setup)
```bash
# Option A: Use batch file (Windows)
SETUP.bat

# Option B: Manual installation
pip install -r requirements.txt
```

**Required packages:**
- tensorflow>=2.13.0
- flask>=2.3.0
- numpy>=1.26.0
- mediapipe>=0.10.0
- scikit-learn>=1.3.0
- opencv-python>=4.9.0.80
- matplotlib>=3.7.0
- tensorflowjs>=4.0.0

### Step 2: Launch the Control Panel
```bash
# Option A: Use batch file (Windows)
START.bat

# Option B: Manual start
python app.py
```

**This will:**
1. Check system status
2. Start Flask server on port 5000
3. Open control panel in browser
4. Display system logs

### Step 3: Train the Model
1. Open control panel at http://localhost:5000
2. Verify dataset status shows "Ready ✓"
3. Click "Train Model" button
4. Monitor training progress in real-time
5. Wait for completion (~10-20 minutes)

**Training will:**
- Load 120,000+ training samples
- Train for up to 100 epochs (with early stopping)
- Save best model automatically
- Generate training history and plots
- Export model in multiple formats (.h5, .tflite, .tfjs)

### Step 4: Evaluate Model Accuracy
1. After training completes, click "Evaluate Model"
2. System will test on 15,000 test samples
3. View accuracy and loss metrics
4. Expected accuracy: 95%+

### Step 5: Play the Game!
1. Click "Launch Game" in control panel
2. Or navigate to http://localhost:5000/game
3. Allow camera access when prompted
4. Show hand gestures to control the car

## 🎯 Gesture Controls

| Gesture | Action | How to Perform |
|---------|--------|----------------|
| 🤚 Open Palm | **Accelerate** | Show open palm facing camera |
| ✊ Closed Fist | **Brake** | Make a tight fist |
| 👈 Left Tilt | **Steer Left** | Tilt hand to the left |
| 👉 Right Tilt | **Steer Right** | Tilt hand to the right |
| 🖐️ Neutral | **No Action** | Relaxed hand position |

## 📊 System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    CONTROL PANEL (Flask)                  │
│                  http://localhost:5000                    │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐    │
│  │   Dataset   │  │    Model     │  │    Game     │    │
│  │  Management │  │   Training   │  │  Interface  │    │
│  └─────────────┘  └──────────────┘  └─────────────┘    │
│                                                           │
└──────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌────────┐          ┌─────────┐         ┌──────────┐
    │  Data  │          │ Models  │         │   Web    │
    │  Files │          │  Files  │         │  Assets  │
    └────────┘          └─────────┘         └──────────┘
```

## 📁 File Structure Overview

### Core Files
- **`app.py`** - Main Flask server with API endpoints
- **`check_status.py`** - Quick system status checker
- **`START.bat`** - Windows startup script
- **`SETUP.bat`** - Dependency installer

### Dataset Files (`data/`)
- **`gestures_train.npz`** - 120,000 training samples ✅
- **`gestures_val.npz`** - 15,000 validation samples ✅
- **`gestures_test.npz`** - 15,000 test samples ✅
- **`label_map.json`** - Class label mappings ✅
- **`norm_params.json`** - Normalization parameters ✅

### Training Files (`training/`)
- **`train.py`** - Model training script
  - Loads dataset
  - Creates neural network
  - Trains with callbacks
  - Saves model in multiple formats

### Model Files (`models/`) - Created after training
- **`model.keras.h5`** - Keras model (main format)
- **`model.tflite`** - TensorFlow Lite (mobile/embedded)
- **`model_tfjs/`** - TensorFlow.js (web browser)
- **`training_history.json`** - Training metrics
- **`training_history.png`** - Training plots

### Web Files (`web/`)
- **`index.html`** - Game interface HTML
- **`main.js`** - Game logic and controls
- **`model-loader.js`** - Model loading utilities

### Templates (`templates/`)
- **`control_panel.html`** - Control panel UI

## 🔧 API Endpoints

The control panel provides these API endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Control panel UI |
| `/game` | GET | Game interface |
| `/api/status` | GET | Get system status |
| `/api/training/status` | GET | Get training progress |
| `/api/dataset/generate` | POST | Generate dataset |
| `/api/model/train` | POST | Start training |
| `/api/model/evaluate` | POST | Evaluate model |

## 🎮 Control Panel Features

### 1. Dashboard
- **Dataset Card**: Shows dataset status and sample count
- **Model Card**: Shows training status and accuracy
- **Training Progress Card**: Real-time training metrics

### 2. Actions
- **Generate Dataset**: Create new dataset (if needed)
- **Train Model**: Start model training
- **Evaluate Model**: Test model accuracy
- **Launch Game**: Open game in new tab
- **Refresh Status**: Update all status indicators

### 3. Monitoring
- **Real-time Logs**: View system activity
- **Progress Tracking**: Monitor training epochs
- **Status Indicators**: Visual status badges

## 📈 Expected Performance

### Training Performance
- **Time**: 10-20 minutes (CPU) / 2-5 minutes (GPU)
- **Training Accuracy**: 98%+
- **Validation Accuracy**: 95%+
- **Test Accuracy**: 95%+

### Inference Performance
- **FPS**: 30+ frames per second
- **Latency**: <50ms per prediction
- **Model Size**: ~5MB

### Game Performance
- **Rendering**: 60 FPS (Three.js)
- **Hand Tracking**: 30 FPS (MediaPipe)
- **Overall**: Smooth real-time control

## 🐛 Common Issues & Solutions

### Issue 1: TensorFlow Not Installed
**Error**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
```bash
pip install tensorflow>=2.13.0
```

### Issue 2: Flask Not Installed
**Error**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
```bash
pip install flask>=2.3.0
```

### Issue 3: Camera Not Working
**Error**: Camera access denied in browser

**Solution**:
- Allow camera permissions in browser settings
- Close other apps using the camera
- Try Chrome or Edge browser

### Issue 4: Model Training Fails
**Error**: Out of memory during training

**Solution**:
- Reduce batch size in `training/train.py` (line 139)
- Close other applications
- Use GPU if available

### Issue 5: Low Game FPS
**Problem**: Game runs slowly

**Solution**:
- Reduce MediaPipe model complexity
- Lower camera resolution
- Close unnecessary browser tabs

## 🔍 Verification Checklist

Before playing the game, verify:

- [x] Dataset files exist in `data/` directory
- [ ] TensorFlow is installed (`pip show tensorflow`)
- [ ] Flask is installed (`pip show flask`)
- [ ] Model is trained (check `models/` directory)
- [ ] Control panel starts without errors
- [ ] Camera is accessible in browser

## 📊 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.14+, Linux
- **Python**: 3.8 or higher
- **RAM**: 8GB
- **Storage**: 2GB free space
- **Camera**: Any webcam
- **Browser**: Chrome 90+, Edge 90+, Firefox 88+

### Recommended Requirements
- **RAM**: 16GB
- **GPU**: NVIDIA GPU with CUDA support
- **Camera**: HD webcam (720p+)
- **Internet**: For initial library downloads

## 🎯 Next Steps

### Immediate Actions
1. ✅ Dataset is ready - No action needed
2. ⏳ **Train the model** - Use control panel
3. ✅ Test the game - After training completes

### Optional Enhancements
- Adjust model architecture for better accuracy
- Add more gesture types
- Customize game graphics
- Implement multiplayer mode
- Create mobile version

## 📞 Quick Reference

### Start System
```bash
python app.py
```

### Check Status
```bash
python check_status.py
```

### Train Model Manually
```bash
cd training
python train.py
```

### Access Points
- **Control Panel**: http://localhost:5000
- **Game**: http://localhost:5000/game

## 🎉 Summary

Your gesture-controlled driving game system is **95% ready**!

**What's Done:**
- ✅ Complete codebase
- ✅ Dataset generated (120,000+ samples)
- ✅ Control panel interface
- ✅ Game interface
- ✅ All integrations

**What's Needed:**
- ⏳ Train the model (10-20 minutes)
- ⏳ Test and play!

**To Get Started:**
1. Run `python app.py`
2. Open http://localhost:5000
3. Click "Train Model"
4. Wait for training to complete
5. Click "Launch Game"
6. Have fun! 🎮

---

**Ready to train and play?** Run `python app.py` now!
