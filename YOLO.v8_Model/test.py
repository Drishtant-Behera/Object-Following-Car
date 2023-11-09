#imports
from ultralytics import YOLO
import cv2

#this is my model but you can replace this model with any model and it should still work the same.
model = YOLO('Object-Following-Car/best.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# main loop
while True:
    x = False
    _, img = cap.read()
    results = model.predict(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            x1, y1, x2, y2 = map(int, b)  # Convert coordinates to integers

            # Draw rectangle on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            TrueX = (x2 + x1) / 2
            DifX = x2 - x1

            if len(boxes) == 1:
                x = True

            #comands for car
            if x:
                if DifX >= 200:
                    stop = True
                    print("stop")
                else:
                    stop = False

                if TrueX > 345:
                    print("turn right")
                    right = True
                else:
                    right = False

                if TrueX < 295:
                    print("turn left")
                    left = True
                else:
                    left = False

                if 295 < TrueX < 345:
                    print("move forward")
                    forward = True
                else:
                    forward = False

        if not x:
            print("locate")
            locate = True
        else:
            locate = False

    cv2.imshow('YOLO V8 Detection', img)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
