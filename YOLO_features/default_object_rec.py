from ultralytics import YOLO
import cv2 as cv 
# import mediapipe as mp 
import numpy as np 


def object_rec():

    model = YOLO("yolov8n.pt")

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("frame not captured")
            break 

        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, (1000, 700))

        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = model(frame, verbose = False)

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])


                # confidence 
                confidence = float(box.conf[0])

                # cls
                cls = int(box.cls[0])

                # label
                label = model.names[cls]


                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

                # text
                cv.putText(frame,
                           f"{label} {confidence:.2f}",
                           (x1, y1 - 10),
                           cv.FONT_HERSHEY_COMPLEX,
                           1,
                           (255, 0, 0),
                           2)
                           

        

        cv.imshow("--", frame)

        k = cv.waitKey(1)

        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()

object_rec()



        

