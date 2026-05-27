import cv2 as cv 
import mediapipe as mp 
import numpy as np 


mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.4,
    min_tracking_confidence = 0.5
)

mp_draw = mp.solutions.drawing_utils 

cap = cv.VideoCapture(0)

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



    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
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


            if hand_label == "Right":
                right_thumb = thumb_tip_coords
                right_index = index_tip_coords

            elif hand_label == "Left":
                left_thumb = thumb_tip_coords
                left_index = index_tip_coords

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS    #for the connection between the points.
            )


    if all([left_index, left_thumb, right_thumb, right_index]):

        points = np.array([left_index, left_thumb, right_thumb, right_index], np.int32)

        cv.polylines(frame, [points], isClosed = True, color = (0, 255, 0), thickness = 3)

        mask_frame = np.zeros((h, w), dtype = np.uint8)

        cv.fillPoly(mask_frame, [points], color = 255)

        inverted_frame = 255 - frame 

        frame = np.where(
            mask_frame[:, :, None] == 255,
            inverted_frame,
            frame
        )

        
        # inv_part = cv.bitwise_and(
        #     inverted_frame,
        #     inverted_frame,
        #     mask=mask_frame
        #         )

        # normal_mask = cv.bitwise_not(mask_frame)

        # normal_part = cv.bitwise_and(
        #     frame,
        #     frame,
        #     mask=normal_mask
        #         )

        # frame = cv.add(inv_part, normal_part)


    cv.imshow("inversion", frame)

    k = cv.waitKey(1)

    if k == 27:
        break


cap.release()
cv.destroyAllWindows()