import cv2
from ultralytics import YOLO
import numpy as np

# Load all YOLOv8 model variants
detection_model = YOLO('yolov8n.pt')
segmentation_model = YOLO('yolov8n-seg.pt')
classification_model = YOLO('yolov8n-cls.pt')
pose_model = YOLO('yolov8n-pose.pt')
obb_model = YOLO('yolov8n-obb.pt')

# Set display size for each screen (smaller)
frame_width = 320
frame_height = 240

def resize_with_label(img, label):
    img_resized = cv2.resize(img, (frame_width, frame_height))
    cv2.putText(img_resized, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img_resized

# Capture video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Plain camera stream
    cam_view = resize_with_label(frame.copy(), "Camera")

    # 2. Detection
    det_result = detection_model(frame, verbose=False)
    det_img = det_result[0].plot()
    det_view = resize_with_label(det_img, "Detection")

    # 3. Segmentation
    seg_result = segmentation_model(frame, verbose=False)
    seg_img = seg_result[0].plot()
    seg_view = resize_with_label(seg_img, "Segmentation")

    # 4. Classification
    cls_result = classification_model(frame, verbose=False)
    cls_img = frame.copy()
    cls_resized = cv2.resize(cls_img, (frame_width, frame_height))
    cv2.putText(cls_resized, "Classification", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if hasattr(cls_result[0], 'probs'):
        cls_label = cls_result[0].names[int(cls_result[0].probs.top1)]
        cv2.putText(cls_resized, f'{cls_label}', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cls_view = cls_resized

    # 5. Pose
    pose_result = pose_model(frame, verbose=False)
    pose_img = pose_result[0].plot()
    pose_view = resize_with_label(pose_img, "Pose")

    # 6. OBB
    obb_result = obb_model(frame, verbose=False)
    obb_img = obb_result[0].plot()
    obb_view = resize_with_label(obb_img, "OBB")

    # Stack into a 2x3 grid
    row1 = np.hstack((cam_view, det_view, seg_view))
    row2 = np.hstack((cls_view, pose_view, obb_view))
    combined = np.vstack((row1, row2))

    # Show the combined output
    cv2.imshow("YOLOv8 Multi-Task Grid", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
