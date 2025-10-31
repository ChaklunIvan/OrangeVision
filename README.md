# OrangeVision - YOLO Object Detection

Simple OpenCV YOLO object detection system optimized for OrangePi Zero 3 and MacBook.

## Features

- Cross-platform support (OrangePi Zero 3 Linux & MacBook)
- Configurable object class detection
- External camera and built-in camera support
- Real-time object detection with bounding boxes
- Optimized performance for ARM devices

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download YOLOv8 model (will auto-download on first run):
```bash
# The script will automatically download yolov8n.pt on first run
# Or manually download: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Usage

### Basic usage:
```bash
python yolo_detector.py
```

### Specify camera:
```bash
python yolo_detector.py --camera 0  # Use camera index 0
```

### Detect specific objects only:
```bash
python yolo_detector.py --classes person car bicycle
```

### Use config file:
```bash
python yolo_detector.py --config config.json
```

### Full options:
```bash
python yolo_detector.py --model yolov8n.pt --camera 1 --confidence 0.6 --classes person car --width 640 --height 480
```

## Configuration

Edit `config.json` to customize:

```json
{
    "model": "yolov8n.pt",
    "confidence": 0.5,
    "classes": ["person", "car", "bicycle", "dog", "cat"],
    "camera": {
        "width": 640,
        "height": 480,
        "fps": 30
    }
}
```

## Available Object Classes

The system can detect 80 COCO classes including:
- person, bicycle, car, motorcycle, airplane, bus, train, truck
- bird, cat, dog, horse, sheep, cow, elephant, bear, zebra
- bottle, cup, fork, knife, spoon, bowl, banana, apple
- chair, couch, bed, dining table, laptop, tv, cell phone
- And many more...

## Platform Optimizations

### OrangePi Zero 3:
- Automatic detection of ARM architecture
- Reduced frame rate (15 FPS) for better performance
- Smaller buffer size to reduce latency
- Optimized resolution settings

### MacBook:
- Higher frame rate (30 FPS)
- Full resolution support
- Better camera compatibility

## Controls

- Press 'q' to quit
- Press 's' to save current frame with detections

## Flight Controller Camera Support

### Using Camera Connected to Flight Controller

If your camera is connected to a flight controller (Pixhawk, ArduPilot, PX4), use the network detector:

```bash
# Auto-scan for flight controller video streams
python network_detector.py --scan

# Use specific stream URL
python network_detector.py --source udp://0.0.0.0:5600

# RTSP stream from IP camera
python network_detector.py --source rtsp://192.168.1.100:554/stream
```

### Common Flight Controller Video Streams:
- **MAVLink UDP**: `udp://0.0.0.0:5600` (most common)
- **ArduPilot**: `udp://0.0.0.0:14550`
- **PX4**: `udp://127.0.0.1:5600`
- **IP Camera**: `rtsp://camera_ip:554/stream`

### Network Stream Configuration:
Edit `flight_controller_config.json` for your specific setup.

## Troubleshooting

1. **Camera not found**: Try different camera indices (0, 1, 2...)
2. **Model not loading**: Ensure ultralytics is installed: `pip install ultralytics`
3. **Poor performance on OrangePi**: Lower resolution or reduce confidence threshold
4. **External camera not detected**: Check USB connection and try `lsusb` on Linux
5. **Flight controller camera**: 
   - Check MAVLink connection: `mavproxy.py --master=/dev/ttyUSB0`
   - Verify video stream: `gst-launch-1.0 udpsrc port=5600 ! ...`
   - Try different ports: 5600, 14550, 14551