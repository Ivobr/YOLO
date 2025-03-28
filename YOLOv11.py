import cv2
import time
from ultralytics import YOLO

# Load YOLOv12n model
model = YOLO("traffic_sign_detector.pt")  # Replace with actual path to the model file

# Load video instead of webcam
video_path = "0"  # Update with your local video file path
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a window with a dynamic size
cv2.namedWindow("YOLOv11 Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv11 Inference", frame_width, frame_height)

frames = 0

# start time
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Extract results and draw boxes
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confidences = result.boxes.conf  # Confidence scores
        class_ids = result.boxes.cls  # Class IDs
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            label = model.names[int(class_ids[i])]
            score = confidences[i].item()
            
            if score > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLOv11 Inference", frame)
    frames += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):  # Adjust delay for video playback speed
        break

# calc time needed to run 
end_time = time.time()
run_time = end_time - start_time
fps = frames/run_time
print(f"time needed to do whole video: {run_time:.2f} seconds")
print(f"Video time = 11 sec")
print(f"The frames are: {frames}")
print(f"The fps is: {fps}")


cap.release()
cv2.destroyAllWindows()
