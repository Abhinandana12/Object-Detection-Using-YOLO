# Real-Time Object Detection with YOLOv5 and Speech Output

This project performs real-time object detection using the YOLOv5 model and announces detected objects using text-to-speech (TTS).

## Features

- Uses a pre-trained YOLOv5s model from Ultralytics for object detection.
- Captures live video from the webcam.
- Draws bounding boxes and labels detected objects with confidence scores.
- Announces newly detected objects via speech to avoid repetition.
- Press `q` to quit the application.

## Requirements

- Python 3.7+
- OpenCV
- PyTorch
- pyttsx3 (for offline text-to-speech)
- `ultralytics/yolov5` repository (loaded automatically via `torch.hub`)

## Installation

1. Clone this repository or download the script.
2. Install dependencies:

```bash
pip install -r requirements.txt
