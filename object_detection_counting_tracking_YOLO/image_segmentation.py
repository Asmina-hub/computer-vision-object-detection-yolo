import cv2
from ultralytics import YOLO
import numpy

model=YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture('/Users/asminanassar/Documents/hands_on_cv/simple_object_detection/multiobject.mp4')

while True:
    ret, frame = cap.read()
    results = model.track(frame, classes=[0],persist=True, verbose= False)  # Track only bottles number from documentation
    for r in results:
        annotated_frame =frame.copy()
        masks = r.masks
        if masks is not None and r.boxes is not None and r.boxes.id is not None:
            segmented = masks.data.numpy()
            boxes = r.boxes.xyxy.numpy()
            ids= r.boxes.id.numpy()
            for i,mask in enumerate(segmented):
                person_id = ids[i]
                x1,y1,x2,y2 = boxes[i]
                mask_resized=cv2.resize(mask.astype('uint8')*255, (frame.shape[1], frame.shape[0]))
                contours,_ = cv2.findContours(mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                color = (0,0,255)
                cv2.drawContours(annotated_frame, contours, -1, color, 2)
                cv2.putText(annotated_frame, f'ID: {person_id}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow('Segmented Frame', annotated_frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

            
    

    
