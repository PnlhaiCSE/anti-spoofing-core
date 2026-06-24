#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script for face anti-spoofing model
"""

import os
import sys
import cv2
import numpy as np
import argparse
import warnings
import time

# Add parent directory to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

# Set default model directory to parent's resources folder
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "resources", "anti_spoof_models")


def check_image(image):
    """Check if image aspect ratio is 3:4"""
    if image is None:
        print("Error: Could not load image!")
        return False
    
    height, width, channel = image.shape
    if width/height != 3/4:
        print(f"Warning: Image aspect ratio is {width/height:.2f}, expected 3:4 (0.75)")
        print("Proceeding anyway...")
    return True


def test_image(image_path, model_dir="./resources/anti_spoof_models", device_id=0):
    """Test a single image"""
    print(f"\n{'='*60}")
    print(f"Testing image: {image_path}")
    print(f"{'='*60}\n")
    
    # Load image
    image = cv2.imread(image_path)
    if not check_image(image):
        return
    
    # Initialize model and image cropper
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    # Get face bounding box
    image_bbox = model_test.get_bbox(image)
    if image_bbox is None or len(image_bbox) == 0:
        print("Error: No face detected in image!")
        return
    
    print(f"Face detected: bbox = {image_bbox}")
    
    # Predict using all models
    prediction = np.zeros((1, 3))
    test_speed = 0
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return
    
    model_files = os.listdir(model_dir)
    if not model_files:
        print(f"Error: No models found in {model_dir}")
        return
    
    print(f"\nTesting with {len(model_files)} model(s)...")
    
    for model_name in model_files:
        if not model_name.endswith('.pth'):
            continue
            
        model_path = os.path.join(model_dir, model_name)
        print(f"\n  Model: {model_name}")
        
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
        }
        
        # Crop image patch
        if scale is None:
            param["scale"] = 1
        
        crop_image = image_cropper.crop(**param)
        cv2.imwrite("debug_crop.jpg", crop_image)
        print(crop_image.shape)
        # Predict
        start_time = time.time()
        prediction_single = model_test.predict(crop_image, model_path)
        test_speed += time.time() - start_time
        
        prediction += prediction_single
        print(f"    Prediction: {prediction_single}")
    
    # Average predictions
    label = np.argmax(prediction) if not np.all(prediction == 0) else 0
    value = prediction[0][label] / len(model_files)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Raw predictions: {prediction[0]}")
    print(f"Averaged predictions: {prediction[0] / len(model_files)}")
    print(f"Predicted class: {label}")
    print(f"Total inference time: {test_speed:.3f}s")
    print(f"Average time per model: {test_speed / len(model_files):.3f}s")
    
    # Class mapping: 0=Real, 1/2=Spoofing
    class_names = {0: "REAL FACE", 1: "SPOOFING/FAKE(1)", 2: "SPOOFING/FAKE(2)"}
    
    if label == 0:
        print(f"\n✓ RESULT: {class_names[label]} (confidence: {value:.4f})")
    else:
        print(f"\n✗ RESULT: {class_names[label]} (confidence: {value:.4f})")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Test face anti-spoofing model")
    parser.add_argument("--image", type=str, required=True, 
                        help="Path to test image")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing model files")
    parser.add_argument("--device_id", type=int, default=0,
                        help="GPU device ID (default: 0)")
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    if not os.path.isabs(args.image):
        # First try from current working directory
        if os.path.exists(args.image):
            args.image = os.path.abspath(args.image)
        else:
            # Then try from PROJECT_ROOT
            args.image = os.path.normpath(os.path.join(PROJECT_ROOT, args.image.lstrip('./')))
            if not os.path.exists(args.image):
                args.image = os.path.normpath(os.path.join(PROJECT_ROOT, args.image))
    
    if not os.path.isabs(args.model_dir):
        if os.path.exists(args.model_dir):
            args.model_dir = os.path.abspath(args.model_dir)
        else:
            args.model_dir = os.path.normpath(os.path.join(PROJECT_ROOT, args.model_dir))
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    test_image(args.image, args.model_dir, args.device_id)


if __name__ == "__main__":
    main()
