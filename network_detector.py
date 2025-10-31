import cv2
import numpy as np
import argparse
import json
import socket
import struct
from yolo_detector import YOLODetector, get_platform_info


class NetworkVideoCapture:
    """Handle various network video stream formats"""
    
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.socket = None
        self.setup_capture()
    
    def setup_capture(self):
        """Setup video capture based on source type"""
        if self.source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
            # Network stream (HTTP, RTSP, RTMP)
            print(f"Connecting to network stream: {self.source}")
            self.cap = cv2.VideoCapture(self.source)
            
        elif self.source.startswith('udp://'):
            # UDP stream
            self.setup_udp_stream()
            
        elif self.source.startswith('tcp://'):
            # TCP stream  
            self.setup_tcp_stream()
            
        else:
            # Try as regular camera index or file
            try:
                camera_index = int(self.source)
                self.cap = cv2.VideoCapture(camera_index)
            except ValueError:
                self.cap = cv2.VideoCapture(self.source)
    
    def setup_udp_stream(self):
        """Setup UDP video stream reception"""
        # Extract host and port from udp://host:port
        url_parts = self.source.replace('udp://', '').split(':')
        host = url_parts[0] if url_parts[0] else '0.0.0.0'
        port = int(url_parts[1]) if len(url_parts) > 1 else 5000
        
        print(f"Setting up UDP stream on {host}:{port}")
        
        # Create UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.settimeout(5.0)
        
        print(f"Listening for UDP video stream on {host}:{port}")
    
    def setup_tcp_stream(self):
        """Setup TCP video stream reception"""
        url_parts = self.source.replace('tcp://', '').split(':')
        host = url_parts[0] if url_parts[0] else 'localhost'
        port = int(url_parts[1]) if len(url_parts) > 1 else 5000
        
        print(f"Connecting to TCP stream at {host}:{port}")
        # Implementation depends on specific TCP protocol used by flight controller
    
    def read(self):
        """Read frame from video source"""
        if self.cap:
            return self.cap.read()
        elif self.socket:
            return self.read_udp_frame()
        else:
            return False, None
    
    def read_udp_frame(self):
        """Read frame from UDP stream"""
        try:
            # Receive data (adjust buffer size as needed)
            data, addr = self.socket.recvfrom(65536)
            
            # Decode JPEG frame (common format)
            if data.startswith(b'\xff\xd8'):  # JPEG magic bytes
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
            
            return False, None
            
        except socket.timeout:
            print("UDP stream timeout")
            return False, None
        except Exception as e:
            print(f"UDP read error: {e}")
            return False, None
    
    def release(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        if self.socket:
            self.socket.close()
    
    def isOpened(self):
        """Check if capture is opened"""
        if self.cap:
            return self.cap.isOpened()
        elif self.socket:
            return True
        return False


def scan_mavlink_cameras():
    """Scan for MAVLink camera streams (common in flight controllers)"""
    potential_streams = []
    
    # Common MAVLink video stream ports
    common_ports = [5600, 5601, 14550, 14551]
    
    for port in common_ports:
        # Try UDP streams on common MAVLink ports
        potential_streams.append(f"udp://0.0.0.0:{port}")
        potential_streams.append(f"udp://127.0.0.1:{port}")
    
    # RTSP streams (common for IP cameras)
    rtsp_urls = [
        "rtsp://192.168.1.100:554/stream",  # Common IP camera
        "rtsp://10.1.1.1:554/stream",       # Common drone camera IP
    ]
    potential_streams.extend(rtsp_urls)
    
    return potential_streams


def detect_flight_controller_camera():
    """Try to detect flight controller camera streams"""
    print("Scanning for flight controller camera streams...")
    
    potential_streams = scan_mavlink_cameras()
    
    for stream_url in potential_streams:
        print(f"Testing: {stream_url}")
        
        try:
            cap = NetworkVideoCapture(stream_url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ Found working stream: {stream_url}")
                    cap.release()
                    return stream_url
            cap.release()
        except Exception as e:
            print(f"✗ Failed {stream_url}: {e}")
    
    print("No flight controller camera streams detected")
    return None


def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection with Network Streams")
    parser.add_argument("--source", help="Video source: camera index, file, or network URL")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--classes", nargs="+", help="Classes to detect")
    parser.add_argument("--scan", action="store_true", help="Scan for flight controller cameras")
    parser.add_argument("--config", help="JSON config file")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            args.model = config.get("model", args.model)
            args.confidence = config.get("confidence", args.confidence)
            args.classes = config.get("classes", args.classes)
    
    # Get platform info
    platform_info = get_platform_info()
    print(f"Platform: {platform_info['system']} {platform_info['machine']}")
    
    # Initialize detector
    detector = YOLODetector(args.model, args.confidence)
    
    # Set selected classes if specified
    if args.classes:
        detector.set_selected_classes(args.classes)
    
    # Determine video source
    video_source = args.source
    
    if args.scan or not video_source:
        # Scan for flight controller camera
        detected_stream = detect_flight_controller_camera()
        if detected_stream:
            video_source = detected_stream
        elif not video_source:
            # Fallback to camera 0
            video_source = "0"
    
    if not video_source:
        print("No video source specified or detected")
        return
    
    print(f"Using video source: {video_source}")
    
    # Initialize video capture
    cap = NetworkVideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return
    
    print("Press 'q' to quit, 's' to save current frame")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Detect objects
            boxes, confidences, class_ids = detector.detect_objects(frame)
            
            # Draw detections
            frame = detector.draw_detections(frame, boxes, confidences, class_ids)
            
            # Add frame info
            info_text = f"Source: {video_source} | Frame: {frame_count} | Objects: {len(boxes)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Network YOLO Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"network_detection_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved frame as {filename}")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()