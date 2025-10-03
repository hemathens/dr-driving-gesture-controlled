import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data():
    """Load training, validation, and test datasets"""
    data_dir = Path("../data")
    
    # Load training data
    train_data = np.load(data_dir / "train.npz")
    X_train, y_train = train_data['X'], train_data['y']
    
    # Load validation data
    val_data = np.load(data_dir / "val.npz")
    X_val, y_val = val_data['X'], val_data['y']
    
    # Load test data
    test_data = np.load(data_dir / "test.npz")
    X_test, y_test = test_data['X'], test_data['y']
    
    # Load label map
    with open(data_dir / "label_map.json", 'r') as f:
        label_map = json.load(f)
    
    # Load normalization parameters
    with open(data_dir / "norm_params.json", 'r') as f:
        norm_params = json.load(f)
    
    # Apply normalization
    X_train = (X_train - norm_params['mean']) / (norm_params['std'] + 1e-8)
    X_val = (X_val - norm_params['mean']) / (norm_params['std'] + 1e-8)
    X_test = (X_test - norm_params['mean']) / (norm_params['std'] + 1e-8)
    
    # Convert to float32
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    
    # Convert labels to one-hot encoding
    num_classes = len(label_map['labels'])
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_map

def create_model(input_shape, num_classes):
    """Create the gesture recognition model"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # First dense layer with batch normalization and dropout
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second dense layer
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third dense layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train():
    # Create models directory
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_map = load_data()
    
    # Model parameters
    input_shape = X_train.shape[1:]  # (63,)
    num_classes = len(label_map['labels'])
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create and compile model
    model = create_model(input_shape, num_classes)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        models_dir / "model.keras.h5",
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save model in different formats
    # 1. Keras format (HDF5)
    model.save(models_dir / "model.keras.h5")
    
    # 2. TensorFlow.js format
    tfjs_dir = models_dir / "model_tfjs"
    tfjs_dir.mkdir(exist_ok=True)
    
    tfjs.converters.save_keras_model(model, str(tfjs_dir))
    
    # 3. TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(models_dir / "model.tflite", 'wb') as f:
        f.write(tflite_model)
    
    # Save training history
    history_dict = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    
    with open(models_dir / "training_history.json", 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(models_dir / "training_history.png")
    
    print("\nTraining completed! Models saved in 'models/' directory.")

if __name__ == "__main__":
    train()
