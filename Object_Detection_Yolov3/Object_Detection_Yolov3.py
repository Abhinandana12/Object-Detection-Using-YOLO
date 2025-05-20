import cv2
import numpy as np

# Load YOLOv3 model
model_weights = "Object_Detection\yolov3\yolov3.weights" 
model_config = "Object_Detection\yolov3\yolov3.cfg"
net = cv2.dnn.readNet(model_weights, model_config)

# Load class labels
with open("Object_Detection\yolov3\coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get output layer names
output_layers = net.getUnconnectedOutLayersNames()

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Get original frame dimensions
    H, W = frame.shape[:2]

    # Create a 4D blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialization
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions
            if confidence > 0.5:
                # Scale bounding box coordinates to original frame size
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                width = int(detection[2] * W)
                height = int(detection[3] * H)

                # Calculate top-left corner
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("YOLOv3 Object Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
