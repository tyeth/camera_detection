#!/usr/bin/env python3
"""
GStreamer Camera Detection Script

This script implements real-time object detection using GStreamer for camera capture
and QAI Hub YOLOv11 for detection. Based on the Jupyter notebook implementation.

Designed for Qualcomm QCM6490 devices with CSI cameras.
"""

import os
import sys
import time
import json
import numpy as np
from threading import Thread
from pathlib import Path

# Computer vision imports
import cv2

# Try to import GStreamer
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    HAVE_GSTREAMER = True
    print("✓ GStreamer libraries imported successfully")
except ImportError as e:
    HAVE_GSTREAMER = False
    print(f"✗ GStreamer not available: {e}")

# Import custom squirrel model weights handler
try:
    from squirrel_model_weights import SquirrelModelManager, setup_squirrel_model
    HAVE_SQUIRREL_MODEL = True
    print("✓ Squirrel model weights handler imported successfully")
except ImportError as e:
    HAVE_SQUIRREL_MODEL = False
    print(f"✗ Squirrel model weights handler not available: {e}")

# Try to import QAI Hub models (fallback)
try:
    import qai_hub_models.models.yolov11_det as yolov11_model
    import torch
    HAVE_QAI = True
    print("✓ QAI Hub models available as fallback")
except ImportError as e:
    HAVE_QAI = False
    print(f"✗ QAI Hub models not available: {e}")

# Try to import GPIO (for Linux devices)
try:
    # For Qualcomm devices, this might be different
    # Using Adafruit Blinka for cross-platform GPIO
    import board
    import digitalio
    HAVE_GPIO = True
    print("✓ GPIO support available (Blinka)")
except ImportError:
    try:
        # Fallback to RPi.GPIO
        import RPi.GPIO as GPIO
        HAVE_GPIO = True
        print("✓ GPIO support available (RPi.GPIO)")
    except ImportError:
        HAVE_GPIO = False
        print("✗ GPIO support not available")

class FrameCapture:
    """GStreamer-based frame capture for CSI cameras"""
    
    def __init__(self, camera_id=0):
        """
        Initialize frame capture
        
        Args:
            camera_id: Camera ID (0 for CSI0, 1 for CSI1)
        """
        self.camera_id = camera_id
        self.pipeline = None
        self.loop = None
        self.thread = None
        self.running = False
        self.latest_frame = None
        self.sink = None
        
        if not HAVE_GSTREAMER:
            raise RuntimeError("GStreamer not available")
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Create pipeline
        self.pipeline = self._create_pipeline()
        if not self.pipeline:
            raise RuntimeError("Failed to create GStreamer pipeline")
        
        # Setup main loop
        self.loop = GLib.MainLoop()
        self.thread = Thread(target=self.run_loop)
        
        # Get sink element
        self.sink = self.pipeline.get_by_name("sink")
        if not self.sink:
            raise RuntimeError("Could not find appsink in pipeline")
        
        # Connect to new-sample signal
        self.sink.connect("new-sample", self.on_new_sample)
        
        print(f"✓ Frame capture initialized for camera {camera_id}")
    
    def _create_pipeline(self):
        """Create GStreamer pipeline for Qualcomm CSI camera"""
        # For Qualcomm QCM6490, use qtiqmmfsrc for CSI cameras
        pipeline_str = (
            f"qtiqmmfsrc camera={self.camera_id} ! "
            "video/x-raw,format=NV12,width=640,height=480,framerate=15/1 ! "
            "videoconvert ! "
            "video/x-raw,format=RGB ! "
            "appsink name=sink emit-signals=true"
        )
        
        print(f"Creating pipeline: {pipeline_str}")
        
        try:
            pipeline = Gst.parse_launch(pipeline_str)
            return pipeline
        except GLib.Error as e:
            print(f"Error creating Qualcomm pipeline: {e}")
            
            # Fallback to v4l2src
            fallback_str = (
                "v4l2src device=/dev/video0 ! "
                "videoconvert ! "
                "video/x-raw,format=RGB,width=640,height=480 ! "
                "appsink name=sink emit-signals=true"
            )
            
            print(f"Trying fallback pipeline: {fallback_str}")
            try:
                pipeline = Gst.parse_launch(fallback_str)
                return pipeline
            except GLib.Error as e2:
                print(f"Error creating fallback pipeline: {e2}")
                return None
    
    def run_loop(self):
        """Run the GLib main loop"""
        self.loop.run()
    
    def start(self):
        """Start the frame capture"""
        # Start the pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to set pipeline to playing state")
        
        self.running = True
        self.thread.start()
        print("✓ Frame capture started")
    
    def stop(self):
        """Stop the frame capture"""
        if self.running:
            self.loop.quit()
            self.pipeline.set_state(Gst.State.NULL)
            self.thread.join()
            self.running = False
            print("✓ Frame capture stopped")
    
    def on_new_sample(self, sink):
        """Callback for new frame samples"""
        # Get the sample
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR
        
        # Get buffer and caps
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        # Map buffer data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            buffer.unmap(map_info)
            return Gst.FlowReturn.ERROR
        
        # Get dimensions from caps
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        
        # Copy buffer data to numpy array
        self.latest_frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        ).copy()
        
        # Clean up
        buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def get_frame(self):
        """Get the latest captured frame"""
        return self.latest_frame

class ObjectDetector:
    """Squirrel-specific object detector using custom weights or QAI Hub YOLOv11"""
    
    def __init__(self, confidence_threshold=0.5, model_name="yolov8n_squirrel"):
        """Initialize the object detector with squirrel-specific weights"""
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self.model_manager = None
        self.model = None
        self.class_names = None
        self.target_classes = ['squirrel', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']
        
        # Try to load squirrel-specific model first
        if HAVE_SQUIRREL_MODEL:
            self._load_squirrel_model()
        elif HAVE_QAI:
            self._load_qai_model()
        else:
            print("✗ No detection models available")
    
    def _load_squirrel_model(self):
        """Load squirrel-specific model weights"""
        try:
            print(f"Loading squirrel-specific model: {self.model_name}")
            self.model_manager = setup_squirrel_model(self.model_name)
            
            if self.model_manager and self.model_manager.is_available():
                self.model = self.model_manager
                model_info = self.model_manager.get_model_info()
                print(f"✓ Squirrel model loaded: {model_info['description']}")
                
                if "actual_classes" in model_info:
                    self.class_names = model_info["actual_classes"]
                    squirrel_available = 'squirrel' in self.class_names
                    print(f"✓ Model classes: {self.class_names}")
                    print(f"✓ Squirrel detection available: {squirrel_available}")
                else:
                    self.class_names = model_info["classes"]
                
                return True
            else:
                print("✗ Failed to load squirrel model, trying fallback...")
                return False
                
        except Exception as e:
            print(f"✗ Error loading squirrel model: {e}")
            return False
    
    def _load_qai_model(self):
        """Load QAI Hub model as fallback"""
        try:
            print("Loading QAI Hub YOLOv11 model as fallback...")
            self.model = yolov11_model.Model.from_pretrained()
            self.model.eval()
            
            # Get class names
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                names_dict = self.model.model.names
                self.class_names = [names_dict[i] for i in sorted(names_dict.keys())]
                print(f"✓ QAI Hub model loaded with {len(self.class_names)} classes")
                print("⚠️  Note: Using general COCO model, not squirrel-specific weights")
            else:
                # COCO class names fallback
                self.class_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
                    # ... truncated for brevity
                ]
                
        except Exception as e:
            print(f"✗ Failed to load QAI Hub model: {e}")
            self.model = None
    
    def detect(self, frame):
        """
        Detect objects in frame using squirrel-specific weights
        
        Returns:
            detections: List of detected objects
            triggers: List of trigger classes found
        """
        if self.model is None or frame is None:
            return [], []
        
        try:
            # Use squirrel-specific model if available
            if self.model_manager and hasattr(self.model_manager, 'detect_squirrels'):
                # Use custom squirrel detection
                detections = self.model_manager.detect_squirrels(frame, self.confidence_threshold)
                
                # Extract triggers (squirrels and other target animals)
                triggers = []
                for detection in detections:
                    class_name = detection['class_name'].lower()
                    
                    # Prioritize actual squirrel detections
                    if detection.get('is_squirrel', False) or class_name == 'squirrel':
                        triggers.append('squirrel')
                    elif class_name in [c.lower() for c in self.target_classes]:
                        triggers.append(class_name)
                
                # Remove duplicates while preserving order
                triggers = list(dict.fromkeys(triggers))
                
                print(f"Squirrel-specific detection: {len(detections)} objects, triggers: {triggers}")
                return detections, triggers
                
            else:
                # Fallback to QAI Hub model detection
                return self._detect_with_qai_hub(frame)
                
        except Exception as e:
            print(f"Detection error: {e}")
            return [], []
    
    def _detect_with_qai_hub(self, frame):
        """Fallback detection using QAI Hub model"""
        # Preprocess
        image = frame.astype(np.float32) / 255.0
        image_resized = cv2.resize(image, (640, 640))
        input_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Postprocess
        boxes, scores, class_ids = outputs
        boxes = boxes[0].detach().cpu().numpy()
        scores = scores[0].detach().cpu().numpy()
        class_ids = class_ids[0].detach().cpu().numpy()
        
        # Filter by confidence
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # Scale boxes to original size
        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 640
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y
        
        # Create detection list
        detections = []
        triggers = []
        
        for i in range(len(boxes)):
            class_id = int(class_ids[i])
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
                confidence = float(scores[i])
                
                detection = {
                    'class_name': class_name,
                    'confidence': confidence,
                    'box': boxes[i].tolist(),
                    'is_squirrel': False  # QAI Hub model doesn't have squirrel class
                }
                print(f"Detected: {detection}")
                detections.append(detection)
                
                # Check if it's a target class (no squirrel in COCO)
                if class_name in self.target_classes:
                    triggers.append(class_name)
        
        print(f"QAI Hub fallback detection: {len(detections)} objects, triggers: {triggers}")
        return detections, triggers
    
    def is_available(self):
        """Check if detector is available"""
        return self.model is not None

class GPIOController:
    """GPIO controller for triggering outputs"""
    
    def __init__(self, simulate=False):
        """Initialize GPIO controller"""
        self.simulate = simulate or not HAVE_GPIO
        self.pins = {}
        self.pin_objects = {}  # For Blinka digital IO objects
        
        if not self.simulate:
            print("✓ GPIO controller initialized")
        else:
            print("✓ GPIO controller initialized (simulation mode)")
    
    def setup_pin(self, trigger_class, pin_number):
        """Setup a GPIO pin for a trigger class"""
        self.pins[trigger_class] = pin_number
        
        if not self.simulate:
            try:
                if 'board' in sys.modules:  # Using Blinka
                    # Convert pin number to board pin
                    pin = getattr(board, f"D{pin_number}", None)
                    if pin:
                        digital_pin = digitalio.DigitalInOut(pin)
                        digital_pin.direction = digitalio.Direction.OUTPUT
                        digital_pin.value = False
                        self.pin_objects[pin_number] = digital_pin
                        print(f"✓ Setup GPIO pin {pin_number} for {trigger_class}")
                else:  # Using RPi.GPIO
                    GPIO.setmode(GPIO.BCM)
                    GPIO.setup(pin_number, GPIO.OUT)
                    GPIO.output(pin_number, GPIO.LOW)
                    print(f"✓ Setup GPIO pin {pin_number} for {trigger_class}")
            except Exception as e:
                print(f"✗ Error setting up GPIO pin {pin_number}: {e}")
        else:
            print(f"[SIM] Setup GPIO pin {pin_number} for {trigger_class}")
    
    def trigger(self, trigger_classes):
        """Trigger GPIO pins for detected classes"""
        for trigger_class in trigger_classes:
            if trigger_class in self.pins:
                pin = self.pins[trigger_class]
                
                if not self.simulate:
                    try:
                        if pin in self.pin_objects:  # Blinka
                            self.pin_objects[pin].value = True
                            time.sleep(0.1)
                            self.pin_objects[pin].value = False
                        else:  # RPi.GPIO
                            GPIO.output(pin, GPIO.HIGH)
                            time.sleep(0.1)
                            GPIO.output(pin, GPIO.LOW)
                        print(f"⚡ Triggered GPIO pin {pin} for {trigger_class}")
                    except Exception as e:
                        print(f"✗ Error triggering pin {pin}: {e}")
                else:
                    print(f"[SIM] Triggered GPIO pin {pin} for {trigger_class}")
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if not self.simulate:
            try:
                if 'GPIO' in sys.modules:
                    GPIO.cleanup()
                # Blinka pins are automatically cleaned up
                print("✓ GPIO cleaned up")
            except:
                pass

class CameraDetectionSystem:
    """Complete camera detection system with squirrel-specific weights"""
    
    def __init__(self, camera_id=0, confidence_threshold=0.5, model_name="yolov8n_squirrel"):
        """Initialize the system"""
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        
        # Initialize components
        self.capture = None
        self.detector = None
        self.gpio = None
        self.running = False
        
        print("Initializing camera detection system with squirrel-specific weights...")
        print(f"Target model: {model_name}")
        
        # Check dependencies
        if not HAVE_GSTREAMER:
            raise RuntimeError("GStreamer required for camera capture")
        
        # Initialize components
        self.capture = FrameCapture(camera_id)
        self.detector = ObjectDetector(confidence_threshold, model_name)
        self.gpio = GPIOController(simulate=not HAVE_GPIO)
        
        # Setup GPIO pins
        self.gpio.setup_pin("squirrel", 17)
        self.gpio.setup_pin("bird", 18)
        self.gpio.setup_pin("cat", 22)
        self.gpio.setup_pin("dog", 23)
        
        if not self.detector.is_available():
            print("⚠️  Detector initialized but may be using fallback model")
        else:
            print("✓ Squirrel detection system initialized successfully")
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.detector and hasattr(self.detector, 'model_manager') and self.detector.model_manager:
            return self.detector.model_manager.get_model_info()
        else:
            return {
                "model_name": "fallback",
                "description": "QAI Hub YOLOv11 (general COCO model)",
                "classes": self.detector.class_names if self.detector else [],
                "note": "Using fallback model - no squirrel-specific weights loaded"
            }
    
    def start(self):
        """Start the detection system"""
        print("Starting camera detection system...")
        self.capture.start()
        time.sleep(2)  # Allow camera to initialize
        self.running = True
        print("✓ System started")
    
    def stop(self):
        """Stop the detection system"""
        print("Stopping camera detection system...")
        self.running = False
        if self.capture:
            self.capture.stop()
        if self.gpio:
            self.gpio.cleanup()
        print("✓ System stopped")
    
    def process_frame(self):
        """Process a single frame"""
        if not self.running:
            return None
        
        # Get frame
        frame = self.capture.get_frame()
        if frame is None:
            return None
        
        # Detect objects
        detections, triggers = self.detector.detect(frame)
        
        # Trigger GPIO for detected objects
        if triggers:
            self.gpio.trigger(triggers)
            print(f"🎯 Detected: {triggers}")
        
        return {
            'frame': frame,
            'detections': detections,
            'triggers': triggers
        }
    
    def run(self, duration=None):
        """Run the detection system"""
        self.start()
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while self.running:
                result = self.process_frame()
                
                if result:
                    frame_count += 1
                    
                    if result['triggers']:
                        print(f"Frame {frame_count}: Found {result['triggers']}")
                    
                    # Check duration
                    if duration and (time.time() - start_time) >= duration:
                        break
                
                time.sleep(0.1)  # 10 FPS processing
                
        except KeyboardInterrupt:
            print("\\nInterrupted by user")
        finally:
            self.stop()
        
        elapsed = time.time() - start_time
        print(f"Processed {frame_count} frames in {elapsed:.1f} seconds ({frame_count/elapsed:.1f} FPS)")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GStreamer Camera Detection System with Squirrel Weights")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (0 or 1)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--duration", type=int, help="Run duration in seconds")
    parser.add_argument("--test", action="store_true", help="Run a quick test")
    parser.add_argument("--model", default="yolov8n_squirrel", 
                       help="Model to use (yolov8n_squirrel, yolov8s_squirrel, custom_squirrel)")
    parser.add_argument("--info", action="store_true", help="Show model info and exit")
    
    args = parser.parse_args()
    
    try:
        # Create system
        system = CameraDetectionSystem(
            camera_id=args.camera,
            confidence_threshold=args.confidence,
            model_name=args.model
        )
        
        if args.info:
            # Show model information
            model_info = system.get_model_info()
            print("\n" + "=" * 60)
            print("MODEL INFORMATION")
            print("=" * 60)
            print(f"Model Name: {model_info.get('model_name', 'Unknown')}")
            print(f"Description: {model_info.get('description', 'No description')}")
            print(f"Classes: {model_info.get('classes', [])}")
            if 'actual_classes' in model_info:
                print(f"Actual Classes: {model_info['actual_classes']}")
            print(f"Device: {model_info.get('device', 'Unknown')}")
            print(f"Loaded: {model_info.get('loaded', False)}")
            if 'note' in model_info:
                print(f"Note: {model_info['note']}")
            return
        
        if args.test:
            print("Running 10-second test with squirrel detection...")
            system.run(duration=10)
        else:
            print("Starting squirrel detection system (Ctrl+C to stop)...")
            print(f"Using model: {args.model}")
            system.run(duration=args.duration)
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
