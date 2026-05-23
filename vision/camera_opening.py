# just opening the camera.
import cv2 as cv 
import mediapipe as mp 


def camera_opening():

    cap = cv.VideoCapture(0)    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    return cap