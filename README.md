# Camera Detection System with Squirrel Model

This project implements a real-time object detection system using GStreamer for camera capture and QAI Hub's YOLOv11 model for detection. It's designed to work on Qualcomm QCM6490 devices with CSI cameras and includes GPIO trigger support.

## 🎯 Features

- **Real-time object detection** using QAI Hub YOLOv11 model
- **GStreamer camera capture** optimized for Qualcomm QCM6490 CSI cameras
- **GPIO trigger support** for hardware activation on detection
- **Cross-platform development** with test images on Windows/Linux
- **Squirrel-focused detection** with support for multiple animal classes
- **Hardware acceleration** support for Qualcomm NPU/GPU

## 📁 Project Structure

```
camera_detection/
├── squirrel_model_info.json          # Model metadata and configuration
├── test_squirrel_model.py             # Test script for model verification
├── gstreamer_camera_detection.py     # Main GStreamer detection system
├── download_squirrel_model.py         # Model download and setup utilities
├── gstreamer_frame_capture.ipynb     # Development notebook (reference)
├── requirements.txt                   # Core Python dependencies
├── gstreamer_requirements.txt         # GStreamer-specific requirements
├── gstreamer_yolo_requirements.txt    # Complete requirements list
├── test_images/                       # Sample test images
├── build/                            # Build outputs
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Environment Setup

First, set up the complete environment:

```bash
# Install all dependencies and download model
python download_squirrel_model.py --setup
```

### 2. Test Model Detection

Test the detection system with sample images:

```bash
# Run basic model test
python test_squirrel_model.py

# Get help and options
python test_squirrel_model.py --help
```

### 3. Camera Detection (Qualcomm QCM6490)

Run real-time camera detection:

```bash
# Quick 10-second test
python gstreamer_camera_detection.py --test

# Continuous detection (Ctrl+C to stop)
python gstreamer_camera_detection.py

# Use camera 1 (CSI1) with lower confidence threshold
python gstreamer_camera_detection.py --camera 1 --confidence 0.3

# Run for specific duration
python gstreamer_camera_detection.py --duration 60
```

## 🛠️ Installation

### Method 1: Automatic Setup (Recommended)

```bash
# Clone or download the project
cd camera_detection

# Run automatic setup
python download_squirrel_model.py --setup
```

### Method 2: Manual Installation

#### Core Dependencies

```bash
# Install core AI/ML packages
pip install qai-hub-models[yolov11-det] torch torchvision opencv-python numpy matplotlib

# Install GPIO support (for Qualcomm QCM6490)
pip install Adafruit_Blinka Adafruit_CircuitPython_BusDevice
```

#### System Dependencies (Qualcomm QCM6490)

```bash
# Install GStreamer and development packages
sudo apt-get update
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3-gi \
    python3-gst-1.0 \
    python3-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    pkg-config
```

#### Requirements Files

You can also install from the provided requirements files:

```bash
# Basic setup
pip install -r requirements.txt

# GStreamer-specific setup
pip install -r gstreamer_requirements.txt

# Complete setup with all optional dependencies
pip install -r gstreamer_yolo_requirements.txt
```

## Key Components

1. **Frame Capture**
   - Windows: Uses test images for development
   - Qualcomm: Uses GStreamer with CSI cameras (CSI0 or CSI1)

2. **Object Detection**
   - Uses QAI Hub's YOLOv11 model
   - Configurable to detect specific classes
   - Adjustable confidence threshold

3. **GPIO Triggering**
   - Activates GPIO pins based on detected objects
   - Maps different object classes to different pins

4. **Debug Options**
   - Toggle for showing/hiding detection labels
   - Option to show all detections or only high-confidence ones
   - Enable/disable visual output for headless operation

## GStreamer Command for Qualcomm

The GStreamer pipeline for CSI camera capture:

```
gst-launch-1.0 -e qtiqmmfsrc camera=0 \
! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 \
! qtic2venc ! h264parse ! mp4mux ! queue \
! filesink location=/home/particle/video_snapshot.mp4 \
--gst-debug=qtiqmmfsrc:LOG
```

For frame-by-frame analysis, we modify this to use an appsink:

```
qtiqmmfsrc camera=0 ! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink emit-signals=true
```

## Deployment Instructions

1. Transfer files to the Qualcomm device:
   ```
   scp -r ./* root@192.168.1.169:/home/particle/camera_detection/
   ```

2. Install dependencies on the Qualcomm device:
   ```
   pip install qai-hub-models[yolov11-det] opencv-python numpy matplotlib
   sudo apt-get install -y gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly python3-gi python3-gst-1.0 python3-rpi.gpio
   ```

3. Update the pipeline to use the correct camera:
   - CSI0: `camera_id=0`
   - CSI1: `camera_id=1`

4. Run with real camera:
   ```
   pipeline = FrameProcessingPipeline(use_test_images=False, camera_id=0)
   ```

## Memory Considerations

1. For better performance on resource-constrained devices:
   - Reduce frame resolution (e.g., 640x480 instead of 1920x1080)
   - Process fewer frames per second (adjust `frame_delay`)
   - Use model optimization/quantization via QAI Hub

2. For memory-intensive operations:
   - Release unused resources immediately
   - Clear matplotlib figures after displaying
   - Consider disabling visual output for headless operation

## GPIO Pin Configuration

Current GPIO pin mapping:
- Squirrel detection: GPIO 17
- Bird detection: GPIO 18
- General animal detection: GPIO 27
- Cat detection: GPIO 22
- Dog detection: GPIO 23

Adjust these based on your specific hardware configuration.
