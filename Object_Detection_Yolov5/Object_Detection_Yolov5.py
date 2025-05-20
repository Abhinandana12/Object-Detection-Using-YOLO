import cv2
import torch
import pyttsx3

# Initialize speech engine
engine = pyttsx3.init()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Store previously spoken labels to avoid repeated speech
spoken_labels = set()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(frame_rgb)

    # Extract detection details
    boxes = results.xyxy[0][:, :4]       # Bounding boxes
    confidences = results.xyxy[0][:, 4]  # Confidence scores
    class_ids = results.xyxy[0][:, 5]    # Class indices

    # Track labels in the current frame
    current_labels = set()

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        if confidence > 0.5:
            x_min, y_min, x_max, y_max = map(int, box)
            label = model.names[int(class_id)]
            current_labels.add(label)

            # Draw bounding box and label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Speak newly detected objects
    for label in current_labels:
        if label not in spoken_labels:
            engine.say(f"I see a {label}")
            engine.runAndWait()
            spoken_labels.add(label)

    # Display frame
    cv2.imshow("Object Detection", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
engine.stop()
