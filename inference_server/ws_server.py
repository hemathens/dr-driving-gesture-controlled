import asyncio
import websockets
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

class GestureServer:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        print("Loading model...")
        model_path = Path("models/model.keras.h5")
        if model_path.exists():
            self.model = tf.keras.models.load_model(str(model_path))
            print("Model loaded successfully")
        else:
            print("Model not found!")
    
    async def handle_client(self, websocket, path):
        print(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'predict':
                    landmarks = np.array(data['landmarks']).reshape(1, -1)
                    
                    if self.model:
                        prediction = self.model.predict(landmarks, verbose=0)
                        gesture_idx = np.argmax(prediction[0])
                        confidence = float(prediction[0][gesture_idx])
                        
                        gestures = ['brake', 'forward', 'left', 'neutral', 'right']
                        
                        response = {
                            'type': 'prediction',
                            'gesture': gestures[gesture_idx],
                            'confidence': confidence,
                            'probabilities': prediction[0].tolist()
                        }
                    else:
                        response = {
                            'type': 'error',
                            'message': 'Model not loaded'
                        }
                    
                    await websocket.send(json.dumps(response))
        
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
    
    async def start(self, host='localhost', port=8765):
        print(f"Starting WebSocket server on ws://{host}:{port}")
        async with websockets.serve(self.handle_client, host, port):
            await asyncio.Future()  # Run forever

if __name__ == "__main__":
    server = GestureServer()
    asyncio.run(server.start())
