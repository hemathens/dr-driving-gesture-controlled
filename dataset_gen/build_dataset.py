import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

def augment_landmarks(landmarks, augmentation_factor=3):
    """Apply data augmentation to landmarks"""
    augmented = [landmarks]
    
    for _ in range(augmentation_factor - 1):
        aug = landmarks.copy()
        
        # Random rotation
        angle = np.random.uniform(-15, 15)
        aug = rotate_landmarks(aug, angle)
        
        # Random scale
        scale = np.random.uniform(0.9, 1.1)
        aug = scale_landmarks(aug, scale)
        
        # Random translation
        tx, ty = np.random.uniform(-0.05, 0.05, 2)
        aug = translate_landmarks(aug, tx, ty)
        
        # Random noise
        noise = np.random.normal(0, 0.01, aug.shape)
        aug = aug + noise
        
        augmented.append(aug)
    
    return augmented

def rotate_landmarks(landmarks, angle):
    """Rotate landmarks around center"""
    landmarks_2d = landmarks.reshape(-1, 3)
    center = landmarks_2d.mean(axis=0)
    
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    for i in range(len(landmarks_2d)):
        x, y = landmarks_2d[i][0] - center[0], landmarks_2d[i][1] - center[1]
        landmarks_2d[i][0] = x * cos_a - y * sin_a + center[0]
        landmarks_2d[i][1] = x * sin_a + y * cos_a + center[1]
    
    return landmarks_2d.flatten()

def scale_landmarks(landmarks, scale):
    """Scale landmarks"""
    landmarks_2d = landmarks.reshape(-1, 3)
    center = landmarks_2d.mean(axis=0)
    landmarks_2d = (landmarks_2d - center) * scale + center
    return landmarks_2d.flatten()

def translate_landmarks(landmarks, tx, ty):
    """Translate landmarks"""
    landmarks_2d = landmarks.reshape(-1, 3)
    landmarks_2d[:, 0] += tx
    landmarks_2d[:, 1] += ty
    return landmarks_2d.flatten()

def build_final_dataset():
    """Combine all data sources and create train/val/test splits"""
    print("Building final dataset...")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Load synthetic data
    try:
        data = np.load("data/synthetic_landmarks.npz")
        landmarks = data['landmarks']
        labels = data['labels']
        
        print(f"Loaded {len(landmarks)} synthetic samples")
        
        # Apply augmentation
        print("Applying augmentation...")
        augmented_landmarks = []
        augmented_labels = []
        
        for i in range(len(landmarks)):
            aug_samples = augment_landmarks(landmarks[i], augmentation_factor=3)
            augmented_landmarks.extend(aug_samples)
            augmented_labels.extend([labels[i]] * len(aug_samples))
        
        landmarks = np.array(augmented_landmarks)
        labels = np.array(augmented_labels)
        
        print(f"After augmentation: {len(landmarks)} samples")
        
        # Encode labels
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels_encoded = np.array([label_to_idx[label] for label in labels])
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            landmarks, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Save splits
        np.savez(
            data_dir / "train.npz",
            X=X_train,
            y=y_train
        )
        
        np.savez(
            data_dir / "val.npz",
            X=X_val,
            y=y_val
        )
        
        np.savez(
            data_dir / "test.npz",
            X=X_test,
            y=y_test
        )
        
        # Save label map
        label_map = {
            'labels': unique_labels,
            'label_to_idx': label_to_idx,
            'num_classes': len(unique_labels)
        }
        
        with open(data_dir / "label_map.json", 'w') as f:
            json.dump(label_map, f, indent=2)
        
        # Calculate normalization parameters
        X_flat = landmarks.reshape(-1, 3)  # Reshape to (N*21, 3)
        mean = np.mean(X_flat, axis=0)
        std = np.std(X_flat, axis=0)
        
        norm_params = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        
        with open(data_dir / "norm_params.json", 'w') as f:
            json.dump(norm_params, f, indent=2)
        
        print("\nDataset statistics:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Number of classes: {len(unique_labels)}")
        print(f"  Features per sample: {landmarks.shape[1]}")
        print("\nDataset built successfully!")
        
    except FileNotFoundError:
        print("Error: synthetic_landmarks.npz not found. Please run synth_landmarks.py first.")
        return

if __name__ == "__main__":
    build_final_dataset()
