import cv2 as cv 
import mediapipe as mp 
import numpy as np 



def multi_feature():
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode = False,
        max_num_hands = 2,
        model_complexity = 0,
        min_detection_confidence = 0.4,
        min_tracking_confidence = 0.5
    )


    mp_draw = mp.solutions.darwing_utils


    cap = cv.VideoCapture(0)

    prev_right_thumb = None
    prev_right_index = None
    prev_left_thumb = None
    prev_left_index = None

    portal_activate = False 
    prev_both_pinched = False 

    pinch_threshold = 40


    while True:
        ret, frame = cap.read()
        