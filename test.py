from ultralytics import YOLO

# Load the pre-trained YOLOv12 model
model = YOLO("traffic_sign_detector.pt")
# model2 = YOLO("yolo11n.pt")

# Perform object detection on the image
results = model("./YOLOtraining/trainfotos/stopbord-e1423477567755.jpg")
# results2 = model2("./YOLOtraining/trainfotos/unnamed.jpg");

# Display the results
results[0].show()
# results2[0].show()
