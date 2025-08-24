# Squirrel Detection Models

This directory contains model weights for squirrel detection.

## Directory Structure

- `custom/` - Place your custom-trained squirrel model weights here
- `downloaded/` - Downloaded pre-trained models

## Supported Model Formats

- **YOLOv8 PyTorch weights** (.pt files)
- **YOLOv11 PyTorch weights** (.pt files)
- **ONNX models** (.onnx files) - for deployment optimization

## Custom Model Requirements

Your custom squirrel model should:

1. **Be trained on squirrel data** - The model should be specifically trained to detect squirrels
2. **Use YOLOv8/v11 architecture** - For compatibility with Ultralytics YOLO
3. **Include 'squirrel' class** - The model should have 'squirrel' as one of its classes
4. **Be in PyTorch format** - .pt file extension

## Using Custom Weights

### Option 1: Place weights in custom directory
```bash
# Copy your custom weights
cp your_squirrel_model.pt models/custom/squirrel_weights.pt

# Use the custom model
python gstreamer_camera_detection.py --model custom_squirrel
```

### Option 2: Train your own model

1. **Collect squirrel images** - Gather diverse squirrel photos
2. **Annotate the data** - Use tools like LabelImg, CVAT, or Roboflow
3. **Train with Ultralytics**:
   ```python
   from ultralytics import YOLO
   
   # Load a pre-trained model
   model = YOLO('yolov8n.pt')
   
   # Train on your squirrel dataset
   model.train(data='squirrel_dataset.yaml', epochs=100)
   
   # Save the trained model
   model.save('models/custom/squirrel_weights.pt')
   ```

### Option 3: Use pre-trained animal models

If you don't have custom squirrel weights, you can use general animal detection models:

```bash
# Download YOLOv8 trained on COCO (includes animals)
python squirrel_model_weights.py --model yolov8n_squirrel --download

# This will use the base YOLOv8 model and filter for animal classes
```

## Model Performance Tips

1. **Use appropriate model size**:
   - `yolov8n` - Fastest, least accurate
   - `yolov8s` - Balanced speed/accuracy  
   - `yolov8m` - Better accuracy, slower
   - `yolov8l` - Best accuracy, slowest

2. **Optimize for your hardware**:
   - Use smaller models (nano/small) for real-time detection
   - Use larger models for higher accuracy when speed isn't critical

3. **Adjust confidence threshold**:
   - Lower threshold (0.2-0.4) for catching more detections
   - Higher threshold (0.6-0.8) for reducing false positives

## Example Dataset Structure

If training your own model, structure your data like this:

```
squirrel_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

## Troubleshooting

**Model not loading?**
- Check file path and permissions
- Verify .pt file is not corrupted
- Ensure PyTorch and Ultralytics are installed

**Poor detection performance?**
- Lower confidence threshold
- Check if model was trained on similar data
- Verify camera image quality

**False positives?**
- Increase confidence threshold
- Retrain model with more negative examples
- Add more diverse training data
