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

    pinch_threshold = 40

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
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # mp.draw_landmarks(frame, 
                #                   hand_landmarks, 
                #                   mp_hands.HAND_CONNECTIONS)
                
                hand_label = hand.classification[0].label

                thumb_tip_landmark = hand_landmarks.landmark[4]
                index_tip_landmark = hand_landmarks.landmark[8]

                thumb_tip_coords = (
                    int(thumb_tip_landmark.x * w),
                    int(thumb_tip_landmark.y * h)
                    )
                
                index_tip_coords = (
                    int(index_tip_landmark.x * w),
                    int(index_tip_landmark.y * h)
                    )
                
                distance = np.hypot(
                    thumb_tip_coords[0] - index_tip_coords[0],
                    thumb_tip_coords[1] - index_tip_coords[1]
                )

                if hand_label == "Right":
                    
                    right_pinch = distance < pinch_threshold

                    if prev_right_thumb is not None and prev_right_index is not None:
                        smooth_thumb = (
                            int(0.5 * prev_right_thumb[0] + 0.5 * thumb_tip_coords[0]), 
                            int(0.5 * prev_right_thumb[1] + 0.5 * thumb_tip_coords[1])
                        )

                        smooth_index = (
                            int(0.5 * prev_right_index[0] + 0.5 * index_tip_coords[0]),
                            int(0.5 * prev_right_index[1] + 0.5 * index_tip_coords[1])
                        )

                    else:
                        smooth_thumb = thumb_tip_coords 
                        smooth_index = index_tip_coords


                    right_thumb = smooth_thumb
                    right_index = smooth_index 

                    prev_right_thumb = smooth_thumb 
                    prev_right_index = smooth_index 



                elif hand_label == "Left":
                    left_pinch = distance < pinch_threshold

                    if prev_left_thumb is not None and prev_left_index is not None:
                        smooth_thumb = (
                            int(0.5 * prev_left_thumb[0] + 0.5 * thumb_tip_coords[0]), 
                            int(0.5 * prev_left_thumb[1] + 0.5 * thumb_tip_coords[1])
                        )

                        smooth_index = (
                            int(0.5 * prev_left_index[0] + 0.5 * index_tip_coords[0]),
                            int(0.5 * prev_left_index[1] + 0.5 * index_tip_coords[1])
                        )

                    else:
                        smooth_thumb = thumb_tip_coords 
                        smooth_index = index_tip_coords


                    left_thumb = smooth_thumb
                    left_index = smooth_index 

                    prev_left_thumb = smooth_thumb 
                    prev_left_index = smooth_index 

                
                mp_draw.draw_lanmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        both_pinched = right_pinch and left_pinch
        if both_pinched and not prev_both_pinched:
            portal_activate = not portal_activate

        prev_both_pinched = both_pinched


        


