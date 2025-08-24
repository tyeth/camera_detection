# GStreamer Frame Capture with Object Detection

This project demonstrates a frame capture and object detection system that works on both Windows (for development) and Qualcomm QCM6490 devices (for deployment).

## Project Structure

- `gstreamer_frame_capture.ipynb`: Main Jupyter notebook for camera frame capture and object detection
- `gstreamer_yolo_requirements.txt`: Requirements for the project
- `test_images/`: Directory containing test images for Windows development

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
