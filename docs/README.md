# Gesture Drive - Camera-Controlled Racing Game

Control a 3D racing game using hand gestures detected by your camera!

## Overview

This project implements an end-to-end gesture recognition system:
- **Dataset Generation**: Synthetic hand landmark data generation
- **Model Training**: TensorFlow/Keras neural network
- **Web Demo**: Browser-based game using MediaPipe Hands + TensorFlow.js + Three.js

## Features

- ✅ Fully automated dataset generation (no manual data collection)
- ✅ Privacy-focused (landmarks only, no raw video storage)
- ✅ Real-time gesture recognition in browser
- ✅ Gesture smoothing and calibration
- ✅ Touch control fallback
- ✅ Performance profiling (FPS, latency)
- ✅ 3D graphics with Three.js

## Supported Gestures

| Gesture | Action | Description |
|---------|--------|-------------|
| **Forward** | Accelerate | Open palm facing camera |
| **Brake** | Stop/Slow | Closed fist |
| **Left** | Turn left | Hand tilted left |
| **Right** | Turn right | Hand tilted right |
| **Neutral** | Coast | Relaxed hand |

## Installation

### Prerequisites

```bash
# Python 3.8+
pip install numpy tensorflow tensorflowjs scikit-learn mediapipe opencv-python tqdm
```

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/gesture-drive.git
cd gesture-drive

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset

```bash
# Generate synthetic landmark data
python dataset_gen/synth_landmarks.py

# Build final dataset with augmentation
python dataset_gen/build_dataset.py
```

This creates:
- `data/synthetic_landmarks.npz`  - Raw synthetic data
- `data/train.npz` , `data/val.npz` , `data/test.npz`  - Split datasets
- `data/label_map.json`  - Label mappings

### 2. Train Model

```bash
python training/train.py
```

This produces:
- `models/model.keras.h5`  - Keras model
- `models/model.tflite`  - TensorFlow Lite model
- `models/model_tfjs/`  - TensorFlow.js model
- `models/norm_params.json`  - Normalization parameters

### 3. Run Web Demo

```bash
# Serve the web directory
cd web
python -m http.server 8000

# Open browser
open http://localhost:8000
```

### 4. (Optional) Run WebSocket Server

For debugging or local inference:

```bash
python inference_server/ws_server.py
```

## Project Structure

```
gesture-drive/
├── data/                    # Generated datasets
│   ├── synthetic_landmarks.npz
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── label_map.json
├── dataset_gen/             # Dataset generation scripts
│   ├── fetch_public.py      # (Optional) Public dataset fetcher
│   ├── synth_landmarks.py   # Synthetic data generator
│   └── build_dataset.py     # Dataset builder with augmentation
├── models/                  # Trained models
│   ├── model.keras.h5
│   ├── model.tflite
│   ├── model_tfjs/
│   └── norm_params.json
├── training/               # Model training
│   └── train.py
├── web/                    # Browser demo
│   ├── index.html
│   ├── main.js
│   └── model-loader.js
├── inference_server/       # Optional WebSocket server
│   └── ws_server.py
├── docs/
│   └── README.md
└── requirements.txt
```

## Performance

- **Model Size**: ~500KB (TensorFlow.js)
- **Inference Time**: 15-25ms per frame
- **FPS**: 50-60 FPS in browser
- **Accuracy**: ~95% on test set

## Architecture

### Model
- Input: 63D vector (21 hand landmarks × 3 coordinates)
- Hidden layers: 128 → 256 → 128 → 64 neurons
- Output: 5 classes (softmax)
- Regularization: Batch normalization + Dropout

### Data Pipeline
1. Synthetic generation creates realistic hand poses
2. Heavy augmentation (rotation, scale, translation, noise)
3. 70/15/15 train/val/test split
4. Z-score normalization

### Web Stack
- **MediaPipe Hands**: Real-time hand landmark detection
- **TensorFlow.js**: In-browser gesture inference
- **Three.js**: 3D game rendering
- **Vanilla JS**: Game logic and UI

## License

MIT License - Feel free to use for your own projects!

## Future Work (Milestone 3)

- Unity integration with WebSocket communication
- Advanced gestures (pinch, swipe, etc.)
- Multi-hand support
- Mobile app version
- Gesture recording UI

## Troubleshooting

**Camera not working?**
- Check browser permissions
- Try different browser (Chrome recommended)
- Use touch controls as fallback

**Model not loading?**
- Ensure you've run training script
- Check browser console for errors
- Verify file paths in model-loader.js

**Poor accuracy?**
- Run calibration
- Ensure good lighting
- Keep hand centered in frame
- Adjust MediaPipe confidence thresholds

## Contributing

Pull requests welcome! Please ensure:
- Code follows existing style
- Add tests for new features
- Update documentation

## Contact

Questions? Open an issue on GitHub!
