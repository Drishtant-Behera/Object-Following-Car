
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator 

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, img = cap.read()
    
    results = model.predict(img)

    for r in results:
        
        annotator = Annotator(img)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
            
            x1, y1, x2, y2 = box.xyxy[0]

            print('Bounding box coordinates:', x1, y1, x2, y2)
          
    img = annotator.result()
    cv2.imshow('YOLO V8 Detection', img)     
    
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()