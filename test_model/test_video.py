#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script for face anti-spoofing model on videos
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


def test_video(video_path, model_dir="./resources/anti_spoof_models", device_id=0):
    """Test a video or camera stream"""
    print(f"\n{'='*60}")
    print(f"Testing video: {video_path}")
    print(f"{'='*60}\n")
    
    # Initialize model and image cropper
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return
    
    model_files = os.listdir(model_dir)
    if not model_files:
        print(f"Error: No models found in {model_dir}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file or camera {video_path}")
        return

    # Check for original aspect ratio limitation
    # In original test.py there's a check for width/height == 3/4.
    # For video, we might resize or just run it directly. 
    # Usually webcam frames are 4:3 (e.g. 640x480) or 16:9. 
    # The get_bbox handles face detection, so we can pass the frame as is.
    
    # We will get model info first to save time in the loop
    models_info = []
    for model_name in model_files:
        if not model_name.endswith('.pth'):
            continue
        model_path = os.path.join(model_dir, model_name)
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        models_info.append({
            "name": model_name,
            "path": model_path,
            "h_input": h_input,
            "w_input": w_input,
            "scale": scale
        })
        
    num_models = len(models_info)
    if num_models == 0:
        print("No valid models (.pth) found.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
            
        frame_count += 1
        
        # Original models often expect 3:4 aspect ratio, you can resize or pad here if needed.
        # But we'll process the original frame first.
        
        # Get face bounding box
        image_bbox = model_test.get_bbox(frame)
        if image_bbox is None or len(image_bbox) == 0:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Video Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Predict using all models
        prediction = np.zeros((1, 3))
        start_time = time.time()
        
        for info in models_info:
            param = {
                "org_img": frame,
                "bbox": image_bbox,
                "scale": info["scale"],
                "out_w": info["w_input"],
                "out_h": info["h_input"],
            }
            if info["scale"] is None:
                param["scale"] = 1
            
            # Crop image patch
            crop_image = image_cropper.crop(**param)
            
            # Predict
            prediction_single = model_test.predict(crop_image, info["path"])
            prediction += prediction_single
            
        test_speed = time.time() - start_time
        
        # Average predictions
        label = np.argmax(prediction) if not np.all(prediction == 0) else 0
        value = prediction[0][label] / num_models
        
        # Label 1 is Real Face, others are Spoofing (based on test.py in root)
        if label == 1:
            result_text = f"REAL FACE ({value:.2f})"
            color = (0, 255, 0) # Green for real
        else:
            result_text = f"SPOOF/FAKE ({value:.2f})"
            color = (0, 0, 255) # Red for spoof

        # Draw bounding box and result on frame
        # image_bbox: [x, y, w, h]
        x, y, w, h = [int(v) for v in image_bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw result text and FPS
        cv2.putText(frame, result_text, (x, max(10, y - 5)), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        fps = 1.0 / test_speed if test_speed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Video Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished processing {frame_count} frames.")


def main():
    parser = argparse.ArgumentParser(description="Test face anti-spoofing model on video")
    parser.add_argument("--video", type=str, required=True, 
                        help="Path to test video or camera index (e.g., 0)")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing model files")
    parser.add_argument("--device_id", type=int, default=0,
                        help="GPU device ID (default: 0)")
    
    args = parser.parse_args()
    
    # Check if video is a camera index
    if args.video.isdigit():
        video_path = int(args.video)
    else:
        # Convert relative paths to absolute
        video_path = args.video
        if not os.path.isabs(video_path):
            if os.path.exists(video_path):
                video_path = os.path.abspath(video_path)
            else:
                video_path = os.path.normpath(os.path.join(PROJECT_ROOT, video_path.lstrip('./')))
                if not os.path.exists(video_path):
                    video_path = os.path.normpath(os.path.join(PROJECT_ROOT, video_path))
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
            
    if not os.path.isabs(args.model_dir):
        if os.path.exists(args.model_dir):
            args.model_dir = os.path.abspath(args.model_dir)
        else:
            args.model_dir = os.path.normpath(os.path.join(PROJECT_ROOT, args.model_dir))
    
    test_video(video_path, args.model_dir, args.device_id)


if __name__ == "__main__":
    main()
