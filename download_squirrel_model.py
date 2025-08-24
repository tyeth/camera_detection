#!/usr/bin/env python3
"""
Squirrel Model Download Script

This script provides utilities for downloading and managing the squirrel detection model.
It includes functions to download from Roboflow and verify model compatibility with QAI Hub.
"""

import os
import sys
import json
import requests
import zipfile
from pathlib import Path
import subprocess
from urllib.parse import urlparse

def load_model_info():
    """Load model information from JSON file"""
    info_file = Path(__file__).parent / "squirrel_model_info.json"
    
    if not info_file.exists():
        print(f"✗ Model info file not found: {info_file}")
        return None
    
    try:
        with open(info_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"✗ Error loading model info: {e}")
        return None

def check_qai_hub_installation():
    """Check if QAI Hub models are properly installed"""
    try:
        import qai_hub_models
        print(f"✓ QAI Hub models version: {qai_hub_models.__version__}")
        return True
    except ImportError:
        print("✗ QAI Hub models not installed")
        return False
    except AttributeError:
        print("✓ QAI Hub models installed (version info not available)")
        return True

def install_qai_hub():
    """Install QAI Hub models with YOLOv11 support"""
    print("Installing QAI Hub models...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "qai-hub-models[yolov11-det]", "--upgrade"
        ])
        print("✓ QAI Hub models installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install QAI Hub models: {e}")
        return False

def download_model_from_qai_hub():
    """Download the YOLOv11 model from QAI Hub"""
    print("Downloading YOLOv11 model from QAI Hub...")
    
    try:
        import qai_hub_models.models.yolov11_det as yolov11_model
        
        # This will download and cache the model
        model = yolov11_model.Model.from_pretrained()
        print("✓ YOLOv11 model downloaded and cached successfully")
        
        # Get model details
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            class_names = model.model.names
            target_classes = ['squirrel', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']
            available_targets = [cls for cls in target_classes if cls in class_names.values()]
            
            print(f"✓ Model supports {len(class_names)} classes")
            print(f"✓ Available target classes: {available_targets}")
            
            if 'squirrel' not in available_targets:
                print("⚠️  Note: 'squirrel' class not found in COCO dataset")
                print("   The model can detect similar animals like 'cat', 'dog', 'bird'")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False

def download_roboflow_dataset(api_key=None, workspace="warren-wiens-dhxt5", project="squirrel-spotter", version=1):
    """
    Download the squirrel dataset from Roboflow
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version
    """
    if not api_key:
        print("⚠️  Roboflow API key required to download dataset")
        print("   Visit https://roboflow.com to create an account and get an API key")
        return False
    
    print(f"Downloading dataset from Roboflow: {workspace}/{project}/v{version}")
    
    try:
        # Import roboflow (user needs to install it separately)
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project)
        dataset = project.version(version).download("yolov11")
        
        print(f"✓ Dataset downloaded to: {dataset.location}")
        return True
        
    except ImportError:
        print("✗ Roboflow package not installed")
        print("   Install with: pip install roboflow")
        return False
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False

def verify_installation():
    """Verify that all components are properly installed"""
    print("=" * 60)
    print("VERIFYING INSTALLATION")
    print("=" * 60)
    
    success = True
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"✗ Python {python_version.major}.{python_version.minor} (requires 3.8+)")
        success = False
    
    # Check QAI Hub models
    if check_qai_hub_installation():
        # Try to load the model
        try:
            import qai_hub_models.models.yolov11_det as yolov11_model
            model = yolov11_model.Model.from_pretrained()
            print("✓ YOLOv11 model loads successfully")
        except Exception as e:
            print(f"✗ Error loading YOLOv11 model: {e}")
            success = False
    else:
        success = False
    
    # Check other dependencies
    dependencies = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch")
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} not installed")
            success = False
    
    # Check GStreamer (optional for camera support)
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
        print(f"✓ GStreamer {Gst.version_string()}")
    except ImportError:
        print("⚠️  GStreamer not available (camera support disabled)")
    
    # Check GPIO support (optional for hardware triggers)
    gpio_available = False
    try:
        import board
        import digitalio
        print("✓ GPIO support (Adafruit Blinka)")
        gpio_available = True
    except ImportError:
        try:
            import RPi.GPIO
            print("✓ GPIO support (RPi.GPIO)")
            gpio_available = True
        except ImportError:
            print("⚠️  GPIO support not available (hardware triggers disabled)")
    
    return success

def setup_environment():
    """Set up the complete environment for squirrel detection"""
    print("=" * 60)
    print("SQUIRREL DETECTION SETUP")
    print("=" * 60)
    
    # Load model info
    model_info = load_model_info()
    if model_info:
        print(f"Model: {model_info['model_info']['name']}")
        print(f"Creator: {model_info['model_info']['creator']}")
        print()
    
    # Install dependencies
    dependencies_needed = []
    
    # Check for Ultralytics (primary for custom weights)
    try:
        import ultralytics
        print("✓ Ultralytics YOLO available")
    except ImportError:
        dependencies_needed.append("ultralytics")
        print("✗ Ultralytics YOLO not installed")
    
    # Check for QAI Hub (fallback)
    if not check_qai_hub_installation():
        dependencies_needed.append("qai-hub-models[yolov11-det]")
    
    # Install missing dependencies
    if dependencies_needed:
        print(f"Installing missing dependencies: {dependencies_needed}")
        try:
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                *dependencies_needed, "--upgrade"
            ])
            print("✓ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install dependencies: {e}")
            return False
    
    # Set up custom weights directory
    print("\nSetting up custom weights directory...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "setup_custom_weights.py", "--setup"
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Custom weights directory set up")
        else:
            print("⚠️  Custom weights setup had issues")
    except Exception as e:
        print(f"⚠️  Could not set up custom weights: {e}")
    
    # Verify installation
    if not verify_installation():
        print("⚠️  Installation verification failed")
        return False
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("✓ All components installed successfully")
    print("\nNEXT STEPS:")
    print("=" * 12)
    print("1. TEST DETECTION:")
    print("   python test_squirrel_model.py")
    print()
    print("2. USE WITH GENERAL ANIMAL DETECTION:")
    print("   python gstreamer_camera_detection.py --test")
    print()
    print("3. ADD CUSTOM SQUIRREL WEIGHTS:")
    print("   python setup_custom_weights.py --instructions")
    print("   # Place your .pt file in models/custom/squirrel_weights.pt")
    print("   python gstreamer_camera_detection.py --model custom_squirrel --test")
    print()
    print("4. CAMERA DETECTION OPTIONS:")
    print("   python gstreamer_camera_detection.py --info  # Show model info")
    print("   python gstreamer_camera_detection.py --model yolov8n_squirrel --camera 1")
    print()
    print("NOTE: The system now prioritizes custom squirrel weights over general models.")
    print("      For best results, train your own squirrel detection model!")
    
    return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Squirrel Model Download and Setup")
    parser.add_argument("--setup", action="store_true", help="Set up complete environment")
    parser.add_argument("--verify", action="store_true", help="Verify installation")
    parser.add_argument("--install-qai", action="store_true", help="Install QAI Hub models")
    parser.add_argument("--download-model", action="store_true", help="Download YOLOv11 model")
    parser.add_argument("--roboflow-key", help="Roboflow API key for dataset download")
    parser.add_argument("--download-dataset", action="store_true", help="Download Roboflow dataset")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # No arguments provided, show help
        parser.print_help()
        print("\nQuick start: python {} --setup".format(sys.argv[0]))
        return
    
    try:
        if args.setup:
            success = setup_environment()
            sys.exit(0 if success else 1)
        
        if args.verify:
            success = verify_installation()
            sys.exit(0 if success else 1)
        
        if args.install_qai:
            success = install_qai_hub()
            sys.exit(0 if success else 1)
        
        if args.download_model:
            success = download_model_from_qai_hub()
            sys.exit(0 if success else 1)
        
        if args.download_dataset:
            if not args.roboflow_key:
                print("✗ Roboflow API key required (--roboflow-key)")
                sys.exit(1)
            
            success = download_roboflow_dataset(api_key=args.roboflow_key)
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
