class GameController {
    constructor() {
        this.hands = null;
        this.camera = null;
        this.gameState = 'stopped'; // stopped, playing, paused
        this.currentGesture = 'neutral';
        this.score = 0;
        this.speed = 0;
        this.carPosition = 0; // -1 to 1 (left to right)
        this.obstacles = [];
        this.performanceStats = { fps: 0, latency: 0, frameCount: 0, lastTime: Date.now() };
        
        // Three.js scene
        this.scene = null;
        this.camera3d = null;
        this.renderer = null;
        this.car = null;
        
        this.initUI();
        this.initMediaPipe();
        this.initThreeJS();
    }
    
    initUI() {
        this.videoElement = document.getElementById('video-feed');
        this.gameCanvas = document.getElementById('game-canvas');
        this.startBtn = document.getElementById('start-btn');
        this.calibrateBtn = document.getElementById('calibrate-btn');
        
        this.startBtn.addEventListener('click', () => this.startGame());
        this.calibrateBtn.addEventListener('click', () => this.calibrate());
        
        this.updateStatus('model', false);
        this.updateStatus('camera', false);
        this.updateStatus('hands', false);
    }
    
    async initMediaPipe() {
        try {
            console.log('Initializing MediaPipe Hands...');
            
            this.hands = new Hands({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
                }
            });
            
            this.hands.setOptions({
                maxNumHands: 1,
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            
            this.hands.onResults((results) => this.onHandsResults(results));
            
            this.updateStatus('hands', true);
            console.log('MediaPipe Hands initialized');
        } catch (error) {
            console.error('MediaPipe initialization error:', error);
        }
    }
    
    async initCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            
            this.videoElement.srcObject = stream;
            this.updateStatus('camera', true);
            
            this.camera = new Camera(this.videoElement, {
                onFrame: async () => {
                    if (this.hands && this.gameState === 'playing') {
                        await this.hands.send({ image: this.videoElement });
                    }
                },
                width: 640,
                height: 480
            });
            
            this.camera.start();
        } catch (error) {
            console.error('Camera initialization error:', error);
            alert('Camera access denied. Using touch controls as fallback.');
        }
    }
    
    initThreeJS() {
        const canvas = this.gameCanvas;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x87CEEB);
        this.scene.fog = new THREE.Fog(0x87CEEB, 10, 50);
        
        // Camera
        this.camera3d = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera3d.position.set(0, 3, 5);
        this.camera3d.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.shadowMap.enabled = true;
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Road
        const roadGeometry = new THREE.PlaneGeometry(6, 50);
        const roadMaterial = new THREE.MeshStandardMaterial({ color: 0x333333 });
        const road = new THREE.Mesh(roadGeometry, roadMaterial);
        road.rotation.x = -Math.PI / 2;
        road.position.y = 0;
        road.receiveShadow = true;
        this.scene.add(road);
        
        // Road lines
        for (let i = -20; i < 30; i += 4) {
            const lineGeometry = new THREE.BoxGeometry(0.2, 0.05, 2);
            const lineMaterial = new THREE.MeshStandardMaterial({ color: 0xFFFFFF });
            const line = new THREE.Mesh(lineGeometry, lineMaterial);
            line.position.set(0, 0.05, i);
            this.scene.add(line);
        }
        
        // Car (player)
        const carGroup = new THREE.Group();
        
        const carBody = new THREE.Mesh(
            new THREE.BoxGeometry(1.2, 0.6, 2),
            new THREE.MeshStandardMaterial({ color: 0xFF0000 })
        );
        carBody.position.y = 0.5;
        carBody.castShadow = true;
        carGroup.add(carBody);
        
        const carTop = new THREE.Mesh(
            new THREE.BoxGeometry(0.9, 0.5, 1.2),
            new THREE.MeshStandardMaterial({ color: 0xFF0000 })
        );
        carTop.position.set(0, 1, 0.2);
        carTop.castShadow = true;
        carGroup.add(carTop);
        
        // Wheels
        const wheelGeometry = new THREE.CylinderGeometry(0.3, 0.3, 0.2, 16);
        const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x111111 });
        
        const wheelPositions = [
            [-0.6, 0.3, 0.8],
            [0.6, 0.3, 0.8],
            [-0.6, 0.3, -0.8],
            [0.6, 0.3, -0.8]
        ];
        
        wheelPositions.forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(...pos);
            wheel.castShadow = true;
            carGroup.add(wheel);
        });
        
        carGroup.position.set(0, 0, 2);
        this.scene.add(carGroup);
        this.car = carGroup;
        
        console.log('Three.js scene initialized');
    }
    
    async onHandsResults(results) {
        const startTime = performance.now();
        
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];
            
            // Flatten landmarks to 63D vector
            const flatLandmarks = [];
            landmarks.forEach(lm => {
                flatLandmarks.push(lm.x, lm.y, lm.z);
            });
            
            // Predict gesture
            const prediction = await window.gestureModel.predict(flatLandmarks);
            this.currentGesture = prediction.gesture;
            
            // Update UI
            this.updateGestureDisplay(prediction.gesture, prediction.confidence);
            
            // Update latency
            const latency = performance.now() - startTime;
            this.performanceStats.latency = latency;
        }
    }
    
    async startGame() {
        // Load model
        if (!window.gestureModel.model) {
            this.startBtn.disabled = true;
            this.startBtn.textContent = 'Loading Model...';
            
            const loaded = await window.gestureModel.loadModel();
            if (loaded) {
                this.updateStatus('model', true);
                this.startBtn.textContent = 'Start Game';
            } else {
                this.startBtn.textContent = 'Model Load Failed';
                return;
            }
        }
        
        // Initialize camera
        if (!this.camera) {
            await this.initCamera();
        }
        
        // Start game
        this.gameState = 'playing';
        this.score = 0;
        this.speed = 0;
        this.carPosition = 0;
        this.obstacles = [];
        
        this.startBtn.textContent = 'Pause';
        this.startBtn.onclick = () => this.togglePause();
        
        this.gameLoop();
    }
    
    togglePause() {
        if (this.gameState === 'playing') {
            this.gameState = 'paused';
            this.startBtn.textContent = 'Resume';
        } else {
            this.gameState = 'playing';
            this.startBtn.textContent = 'Pause';
            this.gameLoop();
        }
    }
    
    calibrate() {
        alert('Calibration: Show each gesture for 3 seconds:\n1. Neutral\n2. Left\n3. Right\n4. Forward\n5. Brake');
        window.gestureModel.resetSmoothing();
    }
    
    gameLoop() {
        if (this.gameState !== 'playing') return;
        
        // Update physics based on gesture
        this.updateGamePhysics();
        
        // Update obstacles
        this.updateObstacles();
        
        // Check collisions
        this.checkCollisions();
        
        // Render scene
        this.renderScene();
        
        // Update UI
        this.updateUI();
        
        // Update FPS
        this.updateFPS();
        
        requestAnimationFrame(() => this.gameLoop());
    }
    
    updateGamePhysics() {
        // Update speed based on gesture
        if (this.currentGesture === 'forward') {
            this.speed = Math.min(this.speed + 0.5, 100);
        } else if (this.currentGesture === 'brake') {
            this.speed = Math.max(this.speed - 2, 0);
        } else {
            this.speed = Math.max(this.speed - 0.2, 0);
        }
        
        // Update car position (steering)
        const steerSpeed = 0.03;
        if (this.currentGesture === 'left') {
            this.carPosition = Math.max(this.carPosition - steerSpeed, -2);
        } else if (this.currentGesture === 'right') {
            this.carPosition = Math.min(this.carPosition + steerSpeed, 2);
        }
        
        // Update score
        this.score += Math.floor(this.speed / 20);
        
        // Update car position in scene
        if (this.car) {
            this.car.position.x = this.carPosition;
        }
    }
    
    updateObstacles() {
        // Spawn obstacles
        if (Math.random() < 0.02 && this.speed > 20) {
            const obstacle = {
                x: (Math.random() - 0.5) * 4,
                z: -20,
                mesh: this.createObstacleMesh()
            };
            this.obstacles.push(obstacle);
            this.scene.add(obstacle.mesh);
        }
        
        // Move obstacles
        const moveSpeed = this.speed / 1000;
        this.obstacles.forEach((obstacle, idx) => {
            obstacle.z += moveSpeed;
            obstacle.mesh.position.z = obstacle.z;
            
            // Remove if behind camera
            if (obstacle.z > 10) {
                this.scene.remove(obstacle.mesh);
                this.obstacles.splice(idx, 1);
            }
        });
    }
    
    createObstacleMesh() {
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshStandardMaterial({ 
            color: Math.random() * 0xffffff 
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.position.y = 0.5;
        return mesh;
    }
    
    checkCollisions() {
        this.obstacles.forEach(obstacle => {
            const dx = this.carPosition - obstacle.x;
            const dz = 2 - obstacle.z;
            const distance = Math.sqrt(dx * dx + dz * dz);
            
            if (distance < 1.5) {
                // Collision!
                this.speed = Math.max(this.speed - 20, 0);
                this.score = Math.max(this.score - 50, 0);
            }
        });
    }
    
    renderScene() {
        if (this.renderer && this.scene && this.camera3d) {
            this.renderer.render(this.scene, this.camera3d);
        }
    }
    
    updateUI() {
        document.getElementById('score').textContent = this.score;
        document.getElementById('speed').textContent = `${Math.floor(this.speed)} km/h`;
    }
    
    updateFPS() {
        this.performanceStats.frameCount++;
        const now = Date.now();
        const delta = now - this.performanceStats.lastTime;
        
        if (delta >= 1000) {
            this.performanceStats.fps = Math.round((this.performanceStats.frameCount * 1000) / delta);
            this.performanceStats.frameCount = 0;
            this.performanceStats.lastTime = now;
            
            document.getElementById('fps').textContent = this.performanceStats.fps;
            document.getElementById('latency').textContent = 
                `${this.performanceStats.latency.toFixed(1)}ms`;
        }
    }
    
    updateGestureDisplay(gesture, confidence) {
        const gestureNameEl = document.getElementById('gesture-name');
        const confidenceEl = document.getElementById('confidence');
        
        gestureNameEl.textContent = gesture.toUpperCase();
        confidenceEl.textContent = `${(confidence * 100).toFixed(0)}%`;
        
        // Color coding
        const colors = {
            left: '#00B4D8',
            right: '#00B4D8',
            forward: '#06FFA5',
            brake: '#EF476F',
            neutral: '#ADB5BD'
        };
        
        gestureNameEl.style.color = colors[gesture] || '#ADB5BD';
    }
    
    updateStatus(component, active) {
        const statusEl = document.getElementById(`${component}-status`);
        if (statusEl) {
            statusEl.className = `status-indicator ${active ? 'active' : 'inactive'}`;
        }
    }
    
    handleTouchControl(gesture) {
        this.currentGesture = gesture;
        this.updateGestureDisplay(gesture, 1.0);
    }
}

// Initialize application
window.addEventListener('DOMContentLoaded', () => {
    window.gameController = new GameController();
});
