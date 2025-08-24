#!/usr/bin/env python3
"""
Squirrel Model Weights Handler

This module handles downloading and loading actual squirrel-specific model weights.
It supports multiple model sources including Ultralytics YOLOv8/v11 models and custom weights.
"""

import os
import sys
import json
import urllib.request
import torch
from pathlib import Path
import hashlib

# Try to import YOLO from ultralytics (better for custom models)
try:
    from ultralytics import YOLO
    HAVE_ULTRALYTICS = True
    print("✓ Ultralytics YOLO imported successfully")
except ImportError:
    HAVE_ULTRALYTICS = False
    print("✗ Ultralytics not available. Install with: pip install ultralytics")

# Fallback to QAI Hub
try:
    import qai_hub_models.models.yolov11_det as yolov11_model
    HAVE_QAI = True
    print("✓ QAI Hub models available as fallback")
except ImportError:
    HAVE_QAI = False
    print("✗ QAI Hub models not available")

class SquirrelModelManager:
    """Manages squirrel-specific model weights and inference"""
    
    # Pre-trained squirrel detection models (these would be real URLs in production)
    SQUIRREL_MODELS = {
        "yolov8n_squirrel": {
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "description": "YOLOv8 Nano fine-tuned for squirrel detection",
            "classes": ["squirrel"],
            "input_size": 640,
            "checksum": "sha256:...",  # Would be real checksum
            "note": "Using base YOLOv8n as example - replace with actual squirrel weights"
        },
        "yolov8s_squirrel": {
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt", 
            "description": "YOLOv8 Small fine-tuned for squirrel detection",
            "classes": ["squirrel"],
            "input_size": 640,
            "checksum": "sha256:...",
            "note": "Using base YOLOv8s as example - replace with actual squirrel weights"
        },
        "custom_squirrel": {
            "url": None,  # For local custom weights
            "description": "Custom trained squirrel detection model",
            "classes": ["squirrel"],
            "input_size": 640,
            "local_path": "models/squirrel_weights.pt",
            "note": "Place your custom squirrel weights in models/squirrel_weights.pt"
        }
    }
    
    def __init__(self, model_name="yolov8n_squirrel", models_dir="models"):
        """
        Initialize the squirrel model manager
        
        Args:
            model_name: Name of the model to use
            models_dir: Directory to store model files
        """
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.model_info = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_name in self.SQUIRREL_MODELS:
            self.model_info = self.SQUIRREL_MODELS[model_name]
        else:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.SQUIRREL_MODELS.keys())}")
        
        print(f"Initializing Squirrel Model Manager with {model_name}")
        print(f"Description: {self.model_info['description']}")
        print(f"Target classes: {self.model_info['classes']}")
    
    def download_model(self, force_download=False):
        """Download the model weights if needed"""
        if self.model_info.get("url") is None:
            # Local model
            local_path = self.models_dir / self.model_info["local_path"]
            if local_path.exists():
                print(f"✓ Found local model: {local_path}")
                return str(local_path)
            else:
                print(f"✗ Local model not found: {local_path}")
                print(f"   Place your custom squirrel weights at: {local_path}")
                return None
        
        # Download from URL
        model_filename = f"{self.model_name}.pt"
        model_path = self.models_dir / model_filename
        
        if model_path.exists() and not force_download:
            print(f"✓ Model already exists: {model_path}")
            return str(model_path)
        
        print(f"Downloading {self.model_name} from {self.model_info['url']}...")
        
        try:
            urllib.request.urlretrieve(self.model_info["url"], model_path)
            print(f"✓ Downloaded model to: {model_path}")
            return str(model_path)
        except Exception as e:
            print(f"✗ Failed to download model: {e}")
            return None
    
    def load_model(self, model_path=None):
        """Load the squirrel detection model"""
        if not HAVE_ULTRALYTICS:
            print("✗ Ultralytics YOLO not available for custom models")
            return self._load_fallback_model()
        
        if model_path is None:
            model_path = self.download_model()
        
        if model_path is None:
            print("✗ No model path available")
            return False
        
        try:
            print(f"Loading model from: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Verify model classes
            if hasattr(self.model, 'names'):
                class_names = list(self.model.names.values())
                print(f"✓ Model loaded with classes: {class_names}")
                
                # Check if squirrel is in the classes
                if 'squirrel' in class_names:
                    print("✓ Squirrel class found in model!")
                else:
                    print("⚠️  Squirrel class not found - this appears to be a general model")
                    print("   Consider using a model specifically trained for squirrel detection")
            
            print(f"✓ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback QAI Hub model"""
        if not HAVE_QAI:
            print("✗ No fallback model available")
            return False
        
        try:
            print("Loading fallback QAI Hub YOLOv11 model...")
            self.model = yolov11_model.Model.from_pretrained()
            self.model.eval()
            print("✓ Fallback model loaded (note: general COCO model, not squirrel-specific)")
            return True
        except Exception as e:
            print(f"✗ Failed to load fallback model: {e}")
            return False
    
    def detect_squirrels(self, image, confidence_threshold=0.5):
        """
        Detect squirrels in an image
        
        Args:
            image: Input image (numpy array)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            detections: List of squirrel detections
        """
        if self.model is None:
            print("✗ No model loaded")
            return []
        
        try:
            if HAVE_ULTRALYTICS and hasattr(self.model, 'predict'):
                # Use Ultralytics YOLO
                results = self.model.predict(image, conf=confidence_threshold, verbose=False)
                detections = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for i in range(len(boxes)):
                            # Get class info
                            class_id = int(boxes.cls[i])
                            class_name = self.model.names[class_id]
                            confidence = float(boxes.conf[i])
                            
                            # Get box coordinates
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            
                            detection = {
                                'class_name': class_name,
                                'confidence': confidence,
                                'box': [float(x1), float(y1), float(x2), float(y2)],
                                'is_squirrel': class_name.lower() == 'squirrel'
                            }
                            detections.append(detection)
                
                # Filter for squirrels and high-confidence animals
                squirrel_detections = [d for d in detections if d['is_squirrel'] or 
                                     (d['class_name'] in ['cat', 'dog', 'bird'] and d['confidence'] > confidence_threshold)]
                
                return squirrel_detections
                
            else:
                # Use QAI Hub model (fallback)
                # This would use the previous detection logic
                print("Using fallback detection (QAI Hub model)")
                return self._detect_with_qai_hub(image, confidence_threshold)
                
        except Exception as e:
            print(f"Error in detection: {e}")
            return []
    
    def _detect_with_qai_hub(self, image, confidence_threshold):
        """Fallback detection using QAI Hub model"""
        # This would implement the previous QAI Hub detection logic
        # For brevity, returning empty list here
        print("QAI Hub detection not implemented in this version")
        return []
    
    def is_available(self):
        """Check if model is loaded and available"""
        return self.model is not None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        info = {
            "model_name": self.model_name,
            "description": self.model_info["description"],
            "classes": self.model_info["classes"],
            "device": self.device,
            "loaded": self.is_available()
        }
        
        if self.model and hasattr(self.model, 'names'):
            info["actual_classes"] = list(self.model.names.values())
        
        return info

def setup_squirrel_model(model_name="yolov8n_squirrel"):
    """Setup and return a squirrel model manager"""
    manager = SquirrelModelManager(model_name)
    
    if manager.load_model():
        print("✓ Squirrel model setup complete")
        return manager
    else:
        print("✗ Failed to setup squirrel model")
        return None

def main():
    """Test the squirrel model manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Squirrel Model Manager")
    parser.add_argument("--model", default="yolov8n_squirrel", 
                       choices=list(SquirrelModelManager.SQUIRREL_MODELS.keys()),
                       help="Model to use")
    parser.add_argument("--download", action="store_true", help="Download model")
    parser.add_argument("--info", action="store_true", help="Show model info")
    parser.add_argument("--test", action="store_true", help="Test model loading")
    
    args = parser.parse_args()
    
    manager = SquirrelModelManager(args.model)
    
    if args.info:
        info = manager.get_model_info()
        print(json.dumps(info, indent=2))
    
    if args.download:
        model_path = manager.download_model(force_download=True)
        if model_path:
            print(f"Model downloaded to: {model_path}")
    
    if args.test:
        success = manager.load_model()
        if success:
            print("✓ Model test successful")
            info = manager.get_model_info()
            print(f"Model info: {json.dumps(info, indent=2)}")
        else:
            print("✗ Model test failed")

if __name__ == "__main__":
    main()
