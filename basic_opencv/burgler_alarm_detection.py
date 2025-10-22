import cv2
import os

# Create absolute path for output directory
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "motion_frames")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

capt = cv2.VideoCapture('basic_opencv/theif_on_cam.mp4')
if not capt.isOpened():
    print("Error: Could not open camera")
    exit()

frames = []
gaps = 5
count = 0

try:
    while True:
        ret, current_frame = capt.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        
        if len(frames) > gaps + 1:
            frames.pop(0)
            
        cv2.putText(current_frame, f"Frames Collected: {count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        if len(frames) > gaps:
            diff = cv2.absdiff(frames[0], frames[-1])
            _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            # Fixed motion detection logic
            for c in contours:
                if cv2.contourArea(c) < 1000:  # Changed > to < for small movements
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Save frame when motion is detected
                cv2.putText(current_frame, "ALERT!!!", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                timestamp = cv2.getTickCount()
                filename = os.path.join(output_dir, 
                                      f"motion_frame_{count:04d}_{timestamp}.jpg")
                cv2.imwrite(filename, current_frame)
                print(f"Motion detected! Saved as: {filename}")

        cv2.imshow("Burglar Alarm System", current_frame)
        count += 1
        
        if cv2.waitKey(30) & 0xFF == 27:  # Press ESC to exit
            print("Program terminated by user")
            break

finally:
    # Cleanup
    capt.release()
    cv2.destroyAllWindows()
    print("Cleanup completed")
