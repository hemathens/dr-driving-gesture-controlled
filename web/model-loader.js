class GestureModel {
    constructor() {
        this.model = null;
        this.normParams = null;
        this.labelMap = null;
        this.gestureHistory = [];
        this.historySize = 5; // Smoothing window
    }
    
    async loadModel() {
        try {
            console.log('Loading TensorFlow.js model...');
            this.model = await tf.loadLayersModel('../models/model_tfjs/model.json');
            console.log('Model loaded successfully');
            
            // Load normalization parameters
            const normResponse = await fetch('../models/norm_params.json');
            this.normParams = await normResponse.json();
            
            // Load label mapping
            const labelResponse = await fetch('../data/label_map.json');
            this.labelMap = await labelResponse.json();
            
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            return false;
        }
    }
    
    normalizeLandmarks(landmarks) {
        // Apply z-score normalization
        const normalized = landmarks.map((value, idx) => {
            const mean = this.normParams.mean[idx];
            const std = this.normParams.std[idx];
            return (value - mean) / (std + 1e-8);
        });
        return normalized;
    }
    
    async predict(landmarks) {
        if (!this.model || !landmarks || landmarks.length !== 63) {
            return { gesture: 'neutral', confidence: 0 };
        }
        
        try {
            // Normalize landmarks
            const normalized = this.normalizeLandmarks(landmarks);
            
            // Create tensor
            const inputTensor = tf.tensor2d([normalized], [1, 63]);
            
            // Predict
            const prediction = await this.model.predict(inputTensor);
            const probs = await prediction.data();
            
            // Get top prediction
            const maxIdx = probs.indexOf(Math.max(...probs));
            const confidence = probs[maxIdx];
            const gesture = this.labelMap.labels[maxIdx];
            
            // Cleanup
            inputTensor.dispose();
            prediction.dispose();
            
            // Apply smoothing
            this.gestureHistory.push({ gesture, confidence });
            if (this.gestureHistory.length > this.historySize) {
                this.gestureHistory.shift();
            }
            
            const smoothedGesture = this.getMostFrequentGesture();
            const avgConfidence = this.getAverageConfidence();
            
            return { gesture: smoothedGesture, confidence: avgConfidence };
        } catch (error) {
            console.error('Prediction error:', error);
            return { gesture: 'neutral', confidence: 0 };
        }
    }
    
    getMostFrequentGesture() {
        if (this.gestureHistory.length === 0) return 'neutral';
        
        const gestureCounts = {};
        this.gestureHistory.forEach(item => {
            gestureCounts[item.gesture] = (gestureCounts[item.gesture] || 0) + 1;
        });
        
        let maxGesture = 'neutral';
        let maxCount = 0;
        for (const [gesture, count] of Object.entries(gestureCounts)) {
            if (count > maxCount) {
                maxCount = count;
                maxGesture = gesture;
            }
        }
        
        return maxGesture;
    }
    
    getAverageConfidence() {
        if (this.gestureHistory.length === 0) return 0;
        const sum = this.gestureHistory.reduce((acc, item) => acc + item.confidence, 0);
        return sum / this.gestureHistory.length;
    }
    
    resetSmoothing() {
        this.gestureHistory = [];
    }
}

window.gestureModel = new GestureModel();
