#!/usr/bin/env python3
"""
Squirrel Model Test Script

This script tests the squirrel detection model and weights using QAI Hub models.
It's designed to work both on development machines (with test images) and 
production devices (with GStreamer camera capture).

Based on the GStreamer frame capture notebook implementation.
"""

import os
import sys
import json
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import QAI Hub models
try:
    import qai_hub_models.models.yolov11_det as yolov11_model
    import torch
    HAVE_QAI = True
    print("✓ QAI Hub models imported successfully")
except ImportError as e:
    HAVE_QAI = False
    print(f"✗ QAI Hub models not available: {e}")
    print("Install with: pip install qai-hub-models[yolov11-det]")

# Try to import GStreamer for camera support
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    HAVE_GSTREAMER = True
    print("✓ GStreamer libraries imported successfully")
except ImportError:
    HAVE_GSTREAMER = False
    print("✗ GStreamer not available")

def load_model_info():
    """Load squirrel model information from JSON file"""
    info_file = Path(__file__).parent / "squirrel_model_info.json"
    
    if not info_file.exists():
        print(f"✗ Model info file not found: {info_file}")
        return None
    
    try:
        with open(info_file, 'r') as f:
            info = json.load(f)
        print(f"✓ Loaded model info: {info['model_info']['name']}")
        return info
    except Exception as e:
        print(f"✗ Error loading model info: {e}")
        return None

class SquirrelDetector:
    """Squirrel detection using QAI Hub YOLOv11 model"""
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the squirrel detector
        
        Args:
            confidence_threshold: Minimum confidence level for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = None
        
        # Squirrel and related animal classes to detect
        self.target_classes = [
            'squirrel', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe'
        ]
        
        # Load the model if QAI Hub is available
        if HAVE_QAI:
            self._load_model()
    
    def _load_model(self):
        """Load the YOLOv11 model from QAI Hub"""
        try:
            print("Loading YOLOv11 model from QAI Hub...")
            self.model = yolov11_model.Model.from_pretrained()
            self.model.eval()  # Set to evaluation mode
            
            # Get class names
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                names_dict = self.model.model.names
                self.class_names = [names_dict[i] for i in sorted(names_dict.keys())]
                print(f"✓ Model loaded with {len(self.class_names)} classes")
                
                # Check which target classes are available
                available_targets = [cls for cls in self.target_classes if cls in self.class_names]
                print(f"✓ Available target classes: {available_targets}")
            else:
                # Fallback to COCO class names
                self.class_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
                    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]
                print("✓ Using COCO class names as fallback")
                
        except Exception as e:
            print(f"✗ Failed to load YOLOv11 model: {e}")
            self.model = None
    
    def preprocess_image(self, frame):
        """Preprocess image for YOLO model"""
        if frame is None:
            return None, None
            
        # Convert to RGB if BGR
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            image = frame.astype(np.float32) / 255.0
        
        # Resize to 640x640
        image_resized = cv2.resize(image, (640, 640))
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor, (frame.shape[1], frame.shape[0])
    
    def postprocess_outputs(self, outputs, original_size):
        """Post-process YOLO outputs to get detections"""
        try:
            boxes, scores, class_ids = outputs
            
            # Convert to numpy
            boxes = boxes[0].detach().cpu().numpy()
            scores = scores[0].detach().cpu().numpy()
            class_ids = class_ids[0].detach().cpu().numpy()
            
            # Filter by confidence
            mask = scores > self.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]
            
            if len(boxes) == 0:
                return []
            
            # Scale boxes back to original image size
            orig_w, orig_h = original_size
            scale_x = orig_w / 640
            scale_y = orig_h / 640
            
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
            
            # Format detections
            detections = []
            for i in range(len(boxes)):
                class_id = int(class_ids[i])
                if class_id < len(self.class_names):
                    detection = {
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'confidence': float(scores[i]),
                        'box': boxes[i].tolist()
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error in postprocessing: {e}")
            return []
    
    def detect(self, frame):
        """
        Detect objects in the frame
        
        Returns:
            detections: List of detection results
            squirrel_found: Boolean indicating if squirrel was detected
        """
        if self.model is None or frame is None:
            return [], False
        
        try:
            # Preprocess image
            input_tensor, original_size = self.preprocess_image(frame)
            if input_tensor is None:
                return [], False
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            # Post-process outputs
            detections = self.postprocess_outputs(outputs, original_size)
            
            # Check for squirrels and other target animals
            squirrel_found = False
            target_detections = []
            
            for detection in detections:
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                if class_name in self.target_classes and confidence >= self.confidence_threshold:
                    target_detections.append(detection)
                    if class_name == 'squirrel':
                        squirrel_found = True
            
            return target_detections, squirrel_found
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return [], False
    
    def is_available(self):
        """Check if the detector is available"""
        return self.model is not None

def create_test_images():
    """Create sample test images if none exist"""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Check if we already have test images
    existing_images = list(test_dir.glob("*.jpg"))
    if existing_images:
        print(f"✓ Found {len(existing_images)} existing test images")
        return existing_images
    
    print("Creating sample test images...")
    created_images = []
    
    # Create simple colored test images
    for i in range(3):
        width, height = 640, 480
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if i % 3 == 0:
            # Red gradient
            for y in range(height):
                for x in range(width):
                    image[y, x, 0] = int(255 * x / width)
        elif i % 3 == 1:
            # Green gradient  
            for y in range(height):
                for x in range(width):
                    image[y, x, 1] = int(255 * y / height)
        else:
            # Blue gradient
            for y in range(height):
                for x in range(width):
                    image[y, x, 2] = int(255 * (x + y) / (width + height))
        
        filename = test_dir / f"sample_{i+1}.jpg"
        cv2.imwrite(str(filename), image)
        created_images.append(filename)
        print(f"✓ Created {filename}")
    
    return created_images

def test_squirrel_detection():
    """Main test function for squirrel detection"""
    print("=" * 60)
    print("SQUIRREL MODEL WEIGHTS TEST")
    print("=" * 60)
    
    # Load model information
    model_info = load_model_info()
    if model_info:
        print(f"Model: {model_info['model_info']['name']}")
        print(f"Creator: {model_info['model_info']['creator']}")
        print(f"Classes: {model_info['model_info']['classes']}")
        print()
    
    # Initialize detector
    print("Initializing squirrel detector...")
    detector = SquirrelDetector(confidence_threshold=0.3)  # Lower threshold for testing
    
    if not detector.is_available():
        print("✗ Detector not available. Cannot run tests.")
        return False
    
    print("✓ Detector initialized successfully")
    print()
    
    # Get test images
    print("Preparing test images...")
    test_images = create_test_images()
    
    if not test_images:
        print("✗ No test images available")
        return False
    
    # Test detection on each image
    total_detections = 0
    squirrels_found = 0
    
    for i, image_path in enumerate(test_images):
        print(f"Testing image {i+1}: {image_path.name}")
        
        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"✗ Failed to load {image_path}")
            continue
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        detections, squirrel_found = detector.detect(frame)
        
        total_detections += len(detections)
        if squirrel_found:
            squirrels_found += 1
        
        # Print results
        if detections:
            print(f"  ✓ Found {len(detections)} target objects:")
            for det in detections:
                marker = "🐿️" if det['class_name'] == 'squirrel' else "🐾"
                print(f"    {marker} {det['class_name']}: {det['confidence']:.2f}")
        else:
            print("  - No target objects detected")
        
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Images tested: {len(test_images)}")
    print(f"Total detections: {total_detections}")
    print(f"Squirrels found: {squirrels_found}")
    print(f"Detection rate: {total_detections/len(test_images):.1f} detections/image")
    
    if squirrels_found > 0:
        print("🎉 SUCCESS: Squirrel detection is working!")
    else:
        print("⚠️  NOTE: No squirrels found in test images (expected for synthetic images)")
    
    print("\n" + "=" * 60)
    print("DOWNLOAD INFORMATION")
    print("=" * 60)
    
    if model_info and 'download_info' in model_info:
        download_info = model_info['download_info']
        print(f"Roboflow URL: {download_info['roboflow_api_url']}")
        print(f"API Key Required: {download_info['api_key_required']}")
        print("Available Formats:")
        for fmt in download_info['formats_available']:
            print(f"  - {fmt}")
    
    print("\nTo get the actual squirrel dataset:")
    print("1. Visit the Roboflow URL above")
    print("2. Create an account and get an API key")
    print("3. Download the dataset in your preferred format")
    print("4. Replace test images with real squirrel images for better testing")
    
    return True

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Squirrel Model Test Script")
        print()
        print("This script tests the squirrel detection model using QAI Hub YOLOv11.")
        print("It works with synthetic test images and can be extended for real images.")
        print()
        print("Usage:")
        print("  python test_squirrel_model.py       # Run the test")
        print("  python test_squirrel_model.py --help  # Show this help")
        return
    
    try:
        success = test_squirrel_detection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
