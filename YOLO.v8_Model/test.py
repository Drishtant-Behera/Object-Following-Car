from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator 

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:

    x = False

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

            TrueX = (x2 + x1) / 2
            DifX = x2 - x1

    
            if len(boxes) == 1:
                x = True
            
            if x == True:
                if DifX >= 200:
                    print ("stop")

                elif TrueX > 345:
                    print ("turn right")
                    
                elif TrueX < 295:
                    print("turn left")
                    
                elif 295 < TrueX < 345:
                    print("move forward")

    if x == False:
        print("locate")

    img = annotator.result()
    cv2.imshow('YOLO V8 Detection', img)
    
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
