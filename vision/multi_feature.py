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

        if not ret:
            print("cannot receive the frame")
            break 

        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, (1000, 700))
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = hands.process(frame)

        h, w, _ = frame.shape 

        right_index = None
        right_thumb = None 
        left_index = None
        left_thumb = None 


        left_pinch = False
        right_pinch = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand in zip(
                results.multi_hand_landmarks, results.multi_handeddness
            ):

                hand_label = hand.classification[0].label

                thumb_tip_landmark = hand_landmarks.landmark[4]
                index_tip_landmark = hand_landmarks.landmark[8]
                middle_tip_landmark = hand_landmarks.landmark[12]
                ring_tip_landmarks = hand_landmarks.landmark[16]
                pinky_tip_landmarks = hand_landmarks.landmark[20]

                thumb_tip_coords = (
                    int(thumb_tip_landmark.x * w),
                    int(thumb_tip_landmark.y * h)
                )

                index_tip_coords = (
                    int(index_tip_landmark.x * w),
                    int(index_tip_landmark.y * h)
                )

                middle_tip_coords = (
                    int(middle_tip_landmark.x * w),
                    int(middle_tip_landmark.y * h)
                )

                ring_tip_coords = (
                    int(ring_tip_landmarks.x * w),
                    int(ring_tip_landmarks.y * h)
                )

                pinky_tip_coords = (    
                    int(pinky_tip_landmarks.x * w),
                    int(pinky_tip_landmarks.y * h)
                )

                distance_index_thumb = np.hypot(
                    thumb_tip_coords[0] - index_tip_coords[0],
                    thumb_tip_coords[1] - index_tip_coords[1]
                )