import cv2
from ultralytics import YOLO

model=YOLO("yolov8n.pt")
cap = cv2.VideoCapture('/Users/asminanassar/Documents/hands_on_cv/simple_object_detection/multiobject.mp4')
while True:
    ret, frame = cap.read()
    results = model(frame, classes=[0])  # Detect only persons, bicycles, and cars
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 