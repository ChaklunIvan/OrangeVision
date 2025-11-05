import cv2
import time
import json
import argparse
from typing import List, Tuple
from yolo_detector import YOLODetector, get_platform_info, find_camera_index
from pymavlink import mavutil
import threading
import queue


class MAVLinkDetector:
    def __init__(self, connection_string: str = '/dev/ttyS5', baudrate: int = 115200):
        """
        Initialize MAVLink detector with UART connection to flight controller
        
        Args:
            connection_string: Serial port for flight controller connection
            baudrate: Serial communication speed
        """
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.mavlink_connection = None
        self.detection_queue = queue.Queue()
        self.running = False
        
        # Connect to flight controller
        self.connect_mavlink()
        
    def connect_mavlink(self):
        """Connect to flight controller via MAVLink"""
        try:
            print(f"Connecting to flight controller: {self.connection_string}:{self.baudrate}")
            self.mavlink_connection = mavutil.mavlink_connection('/dev/ttyS5', baud=115200)
            
            # Wait for heartbeat
            print("Waiting for heartbeat...")
            self.mavlink_connection.wait_heartbeat()
            print("Heartbeat received! MAVLink connection established.")
            
        except Exception as e:
            print(f"Failed to connect to flight controller: {e}")
            print("Will continue with detection only (no MAVLink alerts)")
            self.mavlink_connection = None
    
    def send_detection_alert(self, object_class: str, confidence: float, count: int = 1):
        """Send detection alert to flight controller OSD"""
        if not self.mavlink_connection:
            return
            
        try:
            # Send as STATUSTEXT message (appears on OSD)
            message = f"DETECTED: {object_class.upper()} {confidence:.2f}"
            if count > 1:
                message += f" x{count}"
                
            self.mavlink_connection.mav.statustext_send(
                mavutil.mavlink.MAV_SEVERITY_INFO,
                message.encode()[:50]  # Max 50 chars for STATUSTEXT
            )
            
            # Also send confidence as named value for telemetry
            timestamp = int(time.time() * 1000000) % 4294967295
            param_name = f"OBJ_{object_class[:6].upper()}".encode()[:10]  # Max 10 chars
            
            self.mavlink_connection.mav.named_value_float_send(
                timestamp,
                param_name,
                confidence
            )
            
            print(f"Sent MAVLink alert: {message}")
            
        except Exception as e:
            print(f"Error sending MAVLink message: {e}")
    
    def detection_sender_thread(self):
        """Background thread to send detection alerts"""
        while self.running:
            try:
                # Get detection from queue (with timeout)
                detection_data = self.detection_queue.get(timeout=1.0)
                if detection_data:
                    object_class, confidence, count = detection_data
                    self.send_detection_alert(object_class, confidence, count)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in detection sender thread: {e}")
    
    def queue_detection(self, object_class: str, confidence: float, count: int = 1):
        """Queue detection for sending (non-blocking)"""
        try:
            # Only queue if not already full (prevent spam)
            if self.detection_queue.qsize() < 10:
                self.detection_queue.put((object_class, confidence, count))
        except Exception as e:
            print(f"Error queuing detection: {e}")
    
    def start_detection_sender(self):
        """Start background thread for sending detections"""
        self.running = True
        self.sender_thread = threading.Thread(target=self.detection_sender_thread, daemon=True)
        self.sender_thread.start()
    
    def stop_detection_sender(self):
        """Stop background detection sender"""
        self.running = False
        if hasattr(self, 'sender_thread'):
            self.sender_thread.join(timeout=2.0)


def process_detections_for_mavlink(boxes: List, confidences: List, class_ids: List, 
                                 class_names: List, mavlink_detector: MAVLinkDetector):
    """Process detections and send alerts via MAVLink"""
    if not boxes:
        return
    
    # Group detections by class
    class_detections = {}
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        class_name = class_names[class_id]
        if class_name not in class_detections:
            class_detections[class_name] = []
        class_detections[class_name].append(confidence)
    
    # Send alert for each detected class (with highest confidence)
    for class_name, confidences_list in class_detections.items():
        max_confidence = max(confidences_list)
        count = len(confidences_list)
        
        # Queue detection for MAVLink transmission
        mavlink_detector.queue_detection(class_name, max_confidence, count)


def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection with MAVLink alerts")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--classes", nargs="+", help="Classes to detect")
    parser.add_argument("--config", default="config.json", help="Config file")
    parser.add_argument("--mavlink", default="/dev/ttyS5", help="MAVLink connection string")
    parser.add_argument("--baudrate", type=int, default=115200, help="MAVLink baudrate")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    
    args = parser.parse_args()
    
    # Load config if exists
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
            # Apply Orange Pi optimizations
            if config.get("orangepi_optimizations") and get_platform_info()["is_orangepi"]:
                opt = config["orangepi_optimizations"]
                args.width = opt.get("width", args.width)
                args.height = opt.get("height", args.height)
    except:
        pass
    
    # Get platform info
    platform_info = get_platform_info()
    print(f"Platform: {platform_info['system']} {platform_info['machine']}")
    
    # Initialize YOLO detector
    detector = YOLODetector(args.model, args.confidence)
    
    # Set selected classes if specified
    if args.classes:
        detector.set_selected_classes(args.classes)
    
    # Initialize MAVLink detector
    mavlink_detector = MAVLinkDetector(args.mavlink, args.baudrate)
    mavlink_detector.start_detection_sender()
    
    # Find and initialize camera
    camera_index = args.camera if args.camera is not None else find_camera_index()
    print(f"Using camera index: {camera_index}")
    
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Platform optimizations
    if platform_info["is_orangepi"]:
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Applied Orange Pi optimizations")
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    print("Starting detection with MAVLink alerts...")
    print("Press 'q' to quit")
    
    frame_count = 0
    last_alert_time = 0
    alert_interval = 2.0  # Send alerts every 2 seconds max
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Detect objects
            boxes, confidences, class_ids = detector.detect_objects(frame)
            
            # Send MAVLink alerts (throttled)
            current_time = time.time()
            if boxes and (current_time - last_alert_time) > alert_interval:
                process_detections_for_mavlink(boxes, confidences, class_ids, 
                                             detector.class_names, mavlink_detector)
                last_alert_time = current_time
            
            # Draw detections on frame (for local display)
            frame = detector.draw_detections(frame, boxes, confidences, class_ids)
            
            # Add status info
            status_text = f"Frame: {frame_count} | Objects: {len(boxes)}"
            if mavlink_detector.mavlink_connection:
                status_text += " | MAVLink: Connected"
            else:
                status_text += " | MAVLink: Disconnected"
                
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame (comment out for headless operation)
            #cv2.imshow("MAVLink Object Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        # Cleanup
        mavlink_detector.stop_detection_sender()
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")


if __name__ == "__main__":
    main()