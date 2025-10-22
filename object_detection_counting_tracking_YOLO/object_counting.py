import cv2
from ultralytics import YOLO
import numpy
# points to node over here is if the fps is very high the counting might be inaccurate. also if the object goes out of frame and comes back it will be counted again as a new object.

model=YOLO("yolov8n.pt")
cap = cv2.VideoCapture('/Users/asminanassar/Documents/hands_on_cv/simple_object_detection/count.mp4')
unique_ids = set()
while True:
    ret, frame = cap.read()
    results = model.track(frame, classes=[0],persist=True, verbose= False)  # Track only bottles number from documentation
    annotated_frame = results[0].plot()
    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.numpy()
        for im in ids:
            unique_ids.add(im)
        count = len(unique_ids)
        cv2.putText(annotated_frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLOv8 Object Counting", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    


