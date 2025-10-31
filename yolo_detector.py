import cv2
import numpy as np
import argparse
import json
import platform
import os
from typing import List, Tuple, Optional


class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.net = None
        self.output_layers = None
        self.class_names = []
        self.selected_classes = None
        
        # COCO dataset class names (YOLOv8 default)
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        try:
            # Try to use YOLOv8 with ultralytics
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.class_names = self.coco_classes
                print(f"Loaded YOLOv8 model: {self.model_path}")
            except ImportError:
                # Fallback to OpenCV DNN with YOLO weights
                if self.model_path.endswith('.pt'):
                    print("Ultralytics not available. Please install: pip install ultralytics")
                    print("Or provide .weights and .cfg files for OpenCV DNN")
                    return False
                
                # Load with OpenCV DNN (for .weights files)
                config_path = self.model_path.replace('.weights', '.cfg')
                self.net = cv2.dnn.readNet(self.model_path, config_path)
                layer_names = self.net.getLayerNames()
                self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
                self.class_names = self.coco_classes
                print(f"Loaded YOLO model with OpenCV DNN: {self.model_path}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def set_selected_classes(self, class_names: List[str]):
        """Set which classes to detect"""
        self.selected_classes = []
        for class_name in class_names:
            if class_name in self.class_names:
                self.selected_classes.append(self.class_names.index(class_name))
            else:
                print(f"Warning: Class '{class_name}' not found in model classes")
        
        if self.selected_classes:
            print(f"Selected classes: {[self.class_names[i] for i in self.selected_classes]}")
        else:
            print("No valid classes selected, will detect all classes")
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[List, List, List]:
        """
        Detect objects in frame
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (boxes, confidences, class_ids)
        """
        if hasattr(self, 'model'):
            # Use YOLOv8
            results = self.model(frame, verbose=False)
            
            boxes = []
            confidences = []
            class_ids = []
            
            for result in results:
                for box in result.boxes:
                    if box.conf[0] >= self.confidence_threshold:
                        class_id = int(box.cls[0])
                        
                        # Filter by selected classes if specified
                        if self.selected_classes and class_id not in self.selected_classes:
                            continue
                        
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                        confidences.append(confidence)
                        class_ids.append(class_id)
            
            return boxes, confidences, class_ids
        
        elif self.net is not None:
            # Use OpenCV DNN
            height, width = frame.shape[:2]
            
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence >= self.confidence_threshold:
                        # Filter by selected classes if specified
                        if self.selected_classes and class_id not in self.selected_classes:
                            continue
                        
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
            
            if len(indices) > 0:
                indices = indices.flatten()
                return [boxes[i] for i in indices], [confidences[i] for i in indices], [class_ids[i] for i in indices]
            
            return [], [], []
        
        else:
            print("Model not loaded properly")
            return [], [], []
    
    def draw_detections(self, frame: np.ndarray, boxes: List, confidences: List, class_ids: List) -> np.ndarray:
        """Draw detection boxes on frame"""
        colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x, y, w, h = box
            color = colors[class_id]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"{self.class_names[class_id]}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame


def find_camera_index() -> int:
    """Find available camera index"""
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return i
    return 0


def get_platform_info():
    """Get platform information for camera optimization"""
    system = platform.system()
    machine = platform.machine()
    
    is_orangepi = "arm" in machine.lower() or "aarch64" in machine.lower()
    is_macos = system == "Darwin"
    
    return {
        "system": system,
        "machine": machine,
        "is_orangepi": is_orangepi,
        "is_macos": is_macos
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (auto-detect if not specified)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--classes", nargs="+", help="Classes to detect (e.g., person car)")
    parser.add_argument("--config", help="JSON config file with settings")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    
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
    
    # Find camera
    camera_index = args.camera if args.camera is not None else find_camera_index()
    print(f"Using camera index: {camera_index}")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Optimize for platform
    if platform_info["is_orangepi"]:
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for better performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif platform_info["is_macos"]:
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
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
            cv2.putText(frame, f"Frame: {frame_count} | Objects: {len(boxes)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("YOLO Object Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved frame as {filename}")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()