# Deployment Checklist for Qualcomm QCM6490

## Pre-Deployment Steps (on Windows)
- [ ] Make sure all code changes are saved
- [ ] Run the notebook locally with test images to verify functionality
- [ ] Generate SSH key and test connection to device
- [ ] Verify that all required files are in the project directory

## Files to Transfer
- [ ] gstreamer_frame_capture.ipynb (main notebook)
- [ ] requirements.txt (Python dependencies)
- [ ] README.md (documentation)
- [ ] memory_monitor.py (for performance monitoring)
- [ ] test_images/ (optional for testing)

## Setup on Qualcomm Device
- [ ] Install required system packages:
  ```
  sudo apt-get update
  sudo apt-get install -y gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly python3-gi python3-gst-1.0 python3-rpi.gpio
  ```
- [ ] Install required Python packages:
  ```
  pip install qai-hub-models[yolov11-det] opencv-python numpy matplotlib psutil jupyter
  ```
- [ ] Verify GStreamer is properly configured:
  ```
  gst-inspect-1.0 qtiqmmfsrc
  ```
- [ ] Verify camera connections (CSI0 and CSI1 as needed)
- [ ] Verify GPIO permissions for the user running the code

## Running the Code
- [ ] Start Jupyter notebook:
  ```
  jupyter notebook --ip=0.0.0.0 --no-browser
  ```
- [ ] Connect to Jupyter notebook from your development machine
- [ ] Open gstreamer_frame_capture.ipynb
- [ ] Update the pipeline configuration to use camera_id=0 for CSI0 or camera_id=1 for CSI1
- [ ] Set use_test_images=False when creating the FrameProcessingPipeline
- [ ] Run all cells in sequence
- [ ] Monitor for any errors or warnings

## Troubleshooting
- If GStreamer pipeline fails, check camera connections
- If object detection fails, verify QAI Hub models are installed correctly
- If GPIO triggers don't work, check GPIO permissions and pin configuration
- For memory issues, use memory_monitor.py to track memory usage

## Monitoring and Maintenance
- [ ] Check memory usage during operation
- [ ] Monitor temperature of the device during extended runs
- [ ] Check disk space for accumulated images/logs
- [ ] Set up error logs if running autonomously
