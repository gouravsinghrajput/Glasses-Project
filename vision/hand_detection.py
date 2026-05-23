from camera_opening import camera_opening as camopen 
import cv2 as cv 
import mediapipe as mp 

def hand_detection():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    
    
    hands = mp_hands.Hands(
        max_num_hands = 2,
        min_detection_confidence = 0.6,
        min_tracking_confidence = 0.6
    )

    cap = camopen()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("cannot receive the frame")
            break

        frame = cv.flip(frame, 1)
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  #cv doesn't uses RGB.

        results = hands.process(frame) 

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS)
                

        cv.imshow('hand_detection', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):   #a voice activation to close the hand detection. 
            break 

    cap.release()
    cv.destroyAllWindows()

