#!/bin/bash

# Orange Pi MAVLink Object Detection Runner
# Usage: ./run_mavlink_detection.sh [options]

# Default settings
CAMERA_INDEX=0
MAVLINK_PORT="/dev/ttyUSB0"
BAUDRATE=57600
CONFIDENCE=0.5
CLASSES="person"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --camera)
            CAMERA_INDEX="$2"
            shift 2
            ;;
        --mavlink)
            MAVLINK_PORT="$2"
            shift 2
            ;;
        --baudrate)
            BAUDRATE="$2"
            shift 2
            ;;
        --confidence)
            CONFIDENCE="$2"
            shift 2
            ;;
        --classes)
            CLASSES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --camera INDEX     Camera index (default: 0)"
            echo "  --mavlink PORT     MAVLink serial port (default: /dev/ttyUSB0)"
            echo "  --baudrate RATE    Serial baudrate (default: 57600)"
            echo "  --confidence CONF  Detection confidence (default: 0.5)"
            echo "  --classes CLASSES  Detection classes (default: 'person')"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting MAVLink Object Detection..."
echo "Camera: $CAMERA_INDEX"
echo "MAVLink: $MAVLINK_PORT:$BAUDRATE"
echo "Confidence: $CONFIDENCE"
echo "Classes: $CLASSES"
echo ""

# Check if running on Orange Pi
if [[ $(uname -m) == arm* ]] || [[ $(uname -m) == aarch64 ]]; then
    echo "Detected ARM platform - applying Orange Pi optimizations"
    WIDTH=416
    HEIGHT=416
else
    echo "Detected x86 platform - using standard settings"
    WIDTH=640
    HEIGHT=480
fi

# Run the detection
python3 mavlink_detector.py \
    --camera $CAMERA_INDEX \
    --mavlink $MAVLINK_PORT \
    --baudrate $BAUDRATE \
    --confidence $CONFIDENCE \
    --classes $CLASSES \
    --width $WIDTH \
    --height $HEIGHT \
    --config config.json