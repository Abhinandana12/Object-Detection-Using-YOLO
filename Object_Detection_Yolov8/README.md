# YOLOv8 Multi-Task Video Processing

This project demonstrates running multiple YOLOv8 model variants simultaneously on webcam input, displaying results in a 2x3 grid:

- Plain camera stream
- Object detection (yolov8n.pt)
- Segmentation (yolov8n-seg.pt)
- Classification (yolov8n-cls.pt)
- Pose estimation (yolov8n-pose.pt)
- Oriented bounding boxes (yolov8n-obb.pt)

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO package
- NumPy

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
