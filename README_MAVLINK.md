# MAVLink Object Detection Integration

## Option A Implementation: Parallel Processing + OSD Text Alerts

This implementation processes camera feed on Orange Pi and sends detection alerts to ArduPilot OSD via UART MAVLink.

### Architecture

```
Camera (analog) → ADC → Orange Pi → YOLO Detection → UART MAVLink → Flight Controller → OSD Text → FPV Glasses
Camera (analog) → Flight Controller → Original Video → FPV Glasses
```

### Features

- **Real-time object detection** on Orange Pi Zero 3
- **UART MAVLink communication** to flight controller  
- **OSD text alerts** displayed on FPV feed
- **Detection throttling** to prevent message spam
- **Platform-specific optimizations** for Orange Pi

### Hardware Connections

1. **Camera**: Analog camera → ADC → Orange Pi (USB/CSI)
2. **MAVLink**: Orange Pi UART → Flight Controller UART
3. **Video**: Same camera → Flight Controller (analog feed)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make script executable
chmod +x run_mavlink_detection.sh
```

### Usage

#### Basic Usage
```bash
python3 mavlink_detector.py
```

#### With Options
```bash
python3 mavlink_detector.py \
    --camera 0 \
    --mavlink /dev/ttyUSB0 \
    --baudrate 57600 \
    --confidence 0.5 \
    --classes person car bicycle
```

#### Using Shell Script
```bash
# Default settings
./run_mavlink_detection.sh

# Custom settings
./run_mavlink_detection.sh \
    --camera 0 \
    --mavlink /dev/ttyAMA0 \
    --baudrate 115200 \
    --confidence 0.7 \
    --classes "person car"
```

### Configuration

#### UART Setup (Orange Pi)
```bash
# Enable UART
sudo systemctl enable serial-getty@ttyS0.service

# Check available ports
ls /dev/tty*
```

#### Flight Controller Setup (ArduPilot)
```
# Set serial port for MAVLink
SERIAL2_PROTOCOL = 2 (MAVLink2)
SERIAL2_BAUD = 57600
```

### OSD Display

Detection alerts appear as text on ArduPilot OSD:

```
[Flight data]
DETECTED: PERSON 0.85
[More telemetry]
```

### Messages Sent

1. **STATUSTEXT**: `"DETECTED: PERSON 0.85"` (appears on OSD)
2. **NAMED_VALUE_FLOAT**: Confidence values for telemetry logging

### Performance

#### Orange Pi Zero 3 Optimizations
- **Resolution**: 416x416 (vs 640x480)
- **FPS**: 15 (vs 30)
- **Buffer**: Single frame buffering
- **Model**: YOLOv8 nano for speed

#### Detection Throttling
- **Alert interval**: 2 seconds maximum
- **Queue limit**: 10 messages max
- **Background thread**: Non-blocking MAVLink transmission

### Troubleshooting

#### MAVLink Connection Issues
```bash
# Check serial port
ls -la /dev/ttyUSB0

# Test MAVLink connection
mavproxy.py --master=/dev/ttyUSB0:57600

# Check permissions
sudo usermod -a -G dialout $USER
```

#### Camera Issues
```bash
# List cameras
ls /dev/video*

# Test camera
python3 -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"
```

#### Performance Issues
```bash
# Check CPU usage
htop

# Monitor detection rate
tail -f detection.log
```

### File Structure

```
OrangeVision/
├── mavlink_detector.py        # Main MAVLink detection script
├── yolo_detector.py          # YOLO detection class (unchanged)
├── run_mavlink_detection.sh  # Launch script
├── config.json              # Detection configuration
├── requirements.txt         # Dependencies (with pymavlink)
└── README_MAVLINK.md        # This documentation
```

### Limitations

- **Text-only alerts**: No visual bounding boxes on OSD
- **Static positioning**: Text appears in fixed OSD location
- **Alert throttling**: 2-second intervals to prevent spam
- **UART dependency**: Requires physical UART connection

### Future Enhancements

1. **GPS coordinates**: Add object location in GPS coordinates
2. **Distance estimation**: Calculate object distance from camera
3. **Multiple alerts**: Support multiple simultaneous detections
4. **Log recording**: Save detection events to file
5. **Ground station**: Custom ground station with visual overlays