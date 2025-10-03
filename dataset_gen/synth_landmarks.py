import numpy as np
import json
from pathlib import Path

def generate_synthetic_landmarks(gesture_type, num_samples=1000):
    """Generate synthetic hand landmarks for a specific gesture"""
    landmarks = []
    
    # Base hand structure (21 landmarks: wrist, thumb(4), fingers(4x4))
    # Each landmark has x, y, z coordinates normalized to [0, 1]
    
    for _ in range(num_samples):
        base_landmarks = create_base_hand()
        
        if gesture_type == 'left':
            # Tilted left: wrist higher on right, fingers pointing left
            base_landmarks = tilt_hand(base_landmarks, angle=-20 - np.random.uniform(0, 15))
            
        elif gesture_type == 'right':
            # Tilted right: wrist higher on left, fingers pointing right
            base_landmarks = tilt_hand(base_landmarks, angle=20 + np.random.uniform(0, 15))
            
        elif gesture_type == 'forward':
            # Open palm facing camera
            base_landmarks = open_palm(base_landmarks)
            
        elif gesture_type == 'brake':
            # Closed fist
            base_landmarks = closed_fist(base_landmarks)
            
        elif gesture_type == 'neutral':
            # Relaxed hand, slight curl
            base_landmarks = relaxed_hand(base_landmarks)
        
        # Add natural variation
        base_landmarks = add_noise(base_landmarks, noise_level=0.02)
        
        landmarks.append(base_landmarks.flatten().tolist())
    
    return landmarks

def create_base_hand():
    """Create base hand landmark structure (21 points, 3D)"""
    # Simplified hand model in normalized coordinates
    landmarks = np.zeros((21, 3))
    
    # Wrist (0)
    landmarks[0] = [0.5, 0.7, 0.0]
    
    # Thumb (1-4)
    landmarks[1] = [0.45, 0.65, 0.02]  # CMC
    landmarks[2] = [0.40, 0.58, 0.04]  # MCP
    landmarks[3] = [0.36, 0.52, 0.05]  # IP
    landmarks[4] = [0.33, 0.47, 0.06]  # Tip
    
    # Index finger (5-8)
    landmarks[5] = [0.48, 0.58, 0.01]  # MCP
    landmarks[6] = [0.47, 0.48, 0.02]  # PIP
    landmarks[7] = [0.47, 0.40, 0.03]  # DIP
    landmarks[8] = [0.47, 0.33, 0.04]  # Tip
    
    # Middle finger (9-12)
    landmarks[9] = [0.52, 0.58, 0.01]   # MCP
    landmarks[10] = [0.52, 0.47, 0.02]  # PIP
    landmarks[11] = [0.52, 0.38, 0.03]  # DIP
    landmarks[12] = [0.52, 0.30, 0.04]  # Tip
    
    # Ring finger (13-16)
    landmarks[13] = [0.56, 0.58, 0.01]  # MCP
    landmarks[14] = [0.57, 0.48, 0.02]  # PIP
    landmarks[15] = [0.57, 0.40, 0.03]  # DIP
    landmarks[16] = [0.57, 0.33, 0.04]  # Tip
    
    # Pinky (17-20)
    landmarks[17] = [0.60, 0.60, 0.01]  # MCP
    landmarks[18] = [0.62, 0.52, 0.02]  # PIP
    landmarks[19] = [0.63, 0.46, 0.03]  # DIP
    landmarks[20] = [0.64, 0.41, 0.04]  # Tip
    
    return landmarks

def tilt_hand(landmarks, angle):
    """Rotate hand around z-axis"""
    angle_rad = np.radians(angle)
    center = landmarks[0][:2]  # Wrist as center
    
    for i in range(len(landmarks)):
        # Translate to origin
        x, y = landmarks[i][0] - center[0], landmarks[i][1] - center[1]
        
        # Rotate
        new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        # Translate back
        landmarks[i][0] = new_x + center[0]
        landmarks[i][1] = new_y + center[1]
    
    return landmarks

def open_palm(landmarks):
    """Spread fingers for open palm gesture"""
    for i in range(5, 21):  # All finger joints
        landmarks[i][1] -= 0.03  # Move upward
    return landmarks

def closed_fist(landmarks):
    """Curl fingers for fist gesture"""
    for i in range(5, 21):  # All finger joints
        landmarks[i][1] += 0.05  # Move downward
        landmarks[i][2] += 0.02  # Move forward (toward palm)
    return landmarks

def relaxed_hand(landmarks):
    """Slight natural curl for neutral gesture"""
    for i in range(5, 21):
        landmarks[i][1] += 0.02
    return landmarks

def add_noise(landmarks, noise_level=0.02):
    """Add random noise to simulate natural variation"""
    noise = np.random.normal(0, noise_level, landmarks.shape)
    return landmarks + noise

def save_synthetic_dataset():
    """Generate and save complete synthetic dataset"""
    gestures = ['left', 'right', 'forward', 'brake', 'neutral']
    samples_per_gesture = 2000
    
    all_landmarks = []
    all_labels = []
    
    print("Generating synthetic dataset...")
    
    for gesture in gestures:
        print(f"  Generating {samples_per_gesture} samples for '{gesture}'")
        landmarks = generate_synthetic_landmarks(gesture, samples_per_gesture)
        all_landmarks.extend(landmarks)
        all_labels.extend([gesture] * samples_per_gesture)
    
    # Save as numpy arrays
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    np.savez(
        output_dir / "synthetic_landmarks.npz",
        landmarks=np.array(all_landmarks),
        labels=np.array(all_labels)
    )
    
    # Save metadata
    metadata = {
        'num_samples': len(all_landmarks),
        'num_classes': len(gestures),
        'classes': gestures,
        'landmark_dims': 63,  # 21 landmarks * 3 coordinates
        'source': 'synthetic_generation'
    }
    
    with open(output_dir / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved: {len(all_landmarks)} samples, {len(gestures)} classes")
    return output_dir / "synthetic_landmarks.npz"

if __name__ == "__main__":
    save_synthetic_dataset()
