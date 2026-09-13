# just opening the camera.
import cv2 as cv 
import mediapipe as mp 


def camera_opening():

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv.flip(frame, 1)

        cv.imshow("Camera", frame)

        k = cv.waitKey(1)
        
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()

    # return cap 
