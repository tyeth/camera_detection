#!/usr/bin/env python3
"""
Custom Squirrel Weights Setup

This script helps users set up custom squirrel detection weights.
It provides instructions and utilities for using your own trained squirrel models.
"""

import os
import sys
import json
import shutil
from pathlib import Path
import requests

def setup_custom_weights_directory():
    """Create the models directory structure"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (models_dir / "custom").mkdir(exist_ok=True)
    (models_dir / "downloaded").mkdir(exist_ok=True)
    
    print(f"✓ Created models directory structure:")
    print(f"  - {models_dir}/")
    print(f"  - {models_dir}/custom/")
    print(f"  - {models_dir}/downloaded/")
    
    return models_dir

def create_model_readme(models_dir):
    """Create a README in the models directory"""
    readme_content = """# Squirrel Detection Models

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
"""
    
    readme_path = models_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Created model README: {readme_path}")

def download_example_model():
    """Download an example YOLOv8 model for testing"""
    models_dir = Path("models/downloaded")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download YOLOv8 nano (general purpose)
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    model_path = models_dir / "yolov8n_example.pt"
    
    if model_path.exists():
        print(f"✓ Example model already exists: {model_path}")
        return str(model_path)
    
    print(f"Downloading example YOLOv8 model...")
    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, model_path)
        print(f"✓ Downloaded example model: {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"✗ Failed to download example model: {e}")
        return None

def validate_custom_model(model_path):
    """Validate a custom model file"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"✗ Model file not found: {model_path}")
        return False
    
    print(f"Validating model: {model_path}")
    
    try:
        from ultralytics import YOLO
        
        # Try to load the model
        model = YOLO(str(model_path))
        
        # Check model info
        if hasattr(model, 'names'):
            class_names = list(model.names.values())
            print(f"✓ Model loaded successfully")
            print(f"✓ Classes found: {class_names}")
            
            # Check for squirrel class
            if 'squirrel' in [name.lower() for name in class_names]:
                print(f"✓ Squirrel class found!")
                return True
            else:
                print(f"⚠️  Squirrel class not found in model")
                print(f"   This model will detect: {class_names}")
                print(f"   Consider retraining with squirrel data")
                return True  # Still usable, just not squirrel-specific
        else:
            print(f"⚠️  Could not determine model classes")
            return True
            
    except ImportError:
        print(f"✗ Ultralytics not installed. Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Custom Squirrel Weights Setup")
    parser.add_argument("--setup", action="store_true", help="Set up models directory")
    parser.add_argument("--download-example", action="store_true", help="Download example model")
    parser.add_argument("--validate", help="Validate a custom model file")
    parser.add_argument("--instructions", action="store_true", help="Show training instructions")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # No arguments, show help
        parser.print_help()
        print("\nQuick start:")
        print("  python setup_custom_weights.py --setup")
        print("  python setup_custom_weights.py --download-example")
        return
    
    if args.setup:
        print("Setting up custom weights directory...")
        models_dir = setup_custom_weights_directory()
        create_model_readme(models_dir)
        print("\n✓ Setup complete!")
        print(f"\nTo use custom weights:")
        print(f"1. Place your .pt file in: {models_dir}/custom/squirrel_weights.pt")
        print(f"2. Run: python gstreamer_camera_detection.py --model custom_squirrel")
    
    if args.download_example:
        print("Downloading example model for testing...")
        model_path = download_example_model()
        if model_path:
            print(f"\n✓ Example model ready for testing")
            print(f"Test with: python squirrel_model_weights.py --test --model yolov8n_squirrel")
    
    if args.validate:
        model_path = args.validate
        print(f"Validating custom model: {model_path}")
        is_valid = validate_custom_model(model_path)
        if is_valid:
            print(f"\n✓ Model validation successful")
        else:
            print(f"\n✗ Model validation failed")
    
    if args.instructions:
        print("\n" + "=" * 60)
        print("TRAINING YOUR OWN SQUIRREL MODEL")
        print("=" * 60)
        print("""
1. COLLECT DATA:
   - Take 500-1000+ photos of squirrels in different conditions
   - Include various angles, lighting, backgrounds
   - Add negative examples (no squirrels) for better accuracy

2. ANNOTATE DATA:
   - Use tools like LabelImg, CVAT, or Roboflow
   - Draw bounding boxes around squirrels
   - Label them as "squirrel"

3. PREPARE DATASET:
   - Split into train (70%), val (20%), test (10%)
   - Create YOLO format labels
   - Create dataset.yaml file

4. TRAIN MODEL:
   ```python
   from ultralytics import YOLO
   
   model = YOLO('yolov8n.pt')  # Start with pre-trained model
   model.train(
       data='squirrel_dataset.yaml',
       epochs=100,
       imgsz=640,
       batch=16
   )
   ```

5. VALIDATE AND TEST:
   - Check validation metrics
   - Test on real images
   - Adjust confidence threshold

6. DEPLOY:
   - Save final model as .pt file
   - Place in models/custom/squirrel_weights.pt
   - Use with --model custom_squirrel
        """)

if __name__ == "__main__":
    main()
