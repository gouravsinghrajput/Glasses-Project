import cv2 as cv 
import mediapipe as mp 
import numpy as np 


def magnifier():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode = False,
        max_num_hands = 2,
        model_complexity = 0,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )
    mp_draw = mp.solutions.drawing_utils 

    cap = cv.Video.Capture(0)


    prev_right_thumb =None
    prev_right_index = None
    prev_left_thumb = None  
    prev_left_index = None

    portal_activate = False 
    prev_both_pinched = False

    pinch_treshold = 40

    while True:
        ret, frame = cap.read()
        if not ret:
            print('cannot get the frame')
            break 

        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, (1000, 700))
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 
        h, w, _ = frame.shape

        results = hands.process(frame)

        right_index = None
        right_thumb = None
        left_index = None
        left_thumb = None


        left_pinch = False
        right_pinch = False 


        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand in zip(

            ):
                pass         
