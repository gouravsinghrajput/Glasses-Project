import cv2 as cv 
import mediapipe as mp 
import numpy as np 



def multi_feature():
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        # static_image_mode = False,
        max_num_hands = 2,
        # model_complexity = 0,
        min_detection_confidence = 0.4,
        min_tracking_confidence = 0.3
    )


    mp_draw = mp.solutions.drawing_utils


    cap = cv.VideoCapture(0)

    prev_right_thumb = None
    prev_right_index = None
    prev_right_middle = None 
    prev_right_ring = None 
    prev_right_pinky =None
    prev_left_thumb = None
    prev_left_index = None
    prev_left_middle = None 
    prev_left_ring = None 
    prev_left_pinky =None


    portal_activate = False 
    prev_all_pinched = False 

    pinch_threshold = 35


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
                results.multi_hand_landmarks, results.multi_handedness
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

                distance_middle_thumb = np.hypot(
                    thumb_tip_coords[0] - middle_tip_coords[0],
                    thumb_tip_coords[1] - middle_tip_coords[1]
                )

                distance_ring_thumb = np.hypot(
                    thumb_tip_coords[0] - ring_tip_coords[0],
                    thumb_tip_coords[1] - ring_tip_coords[1]
                )

                distance_pinky_thumb = np.hypot(
                    thumb_tip_coords[0] - pinky_tip_coords[0],
                    thumb_tip_coords[1] - pinky_tip_coords[1]
                )


                if hand_label == "Right":

                    right_pinch = distance_index_thumb < pinch_threshold 
                    right_middle_pinch = distance_middle_thumb < pinch_threshold 
                    right_ring_pinch = distance_ring_thumb < pinch_threshold 
                    right_pinky_pinch = distance_pinky_thumb < pinch_threshold 


                    if prev_right_thumb is not None and prev_right_index is not None and prev_right_middle is not None and prev_right_ring is not None and prev_right_pinky is not None:
                        
                        smooth_thumb = (
                            int(0.5 * prev_right_thumb[0] + 0.5 * thumb_tip_coords[0]),
                            int(0.5 * prev_right_thumb[1] + 0.5 * thumb_tip_coords[1])
                        )

                        smooth_index = (
                            int(0.5 * prev_right_index[0] + 0.5 * index_tip_coords[0]),
                            int(0.5 * prev_right_index[1] + 0.5 * index_tip_coords[1])
                        )

                        smooth_middle = (
                            int(0.5 * prev_right_middle[0] + 0.5 * middle_tip_coords[0]),
                            int(0.5 * prev_right_middle[1] + 0.5 * middle_tip_coords[1])
                        )

                        smooth_ring = (
                            int(0.5 * prev_right_ring[0] + 0.5 * ring_tip_coords[0]),
                            int(0.5 * prev_right_ring[1] + 0.5 * ring_tip_coords[1])
                        )

                        smooth_pinky = (
                            int(0.5 * prev_right_pinky[0] + 0.5 * pinky_tip_coords[0]),
                            int(0.5 * prev_right_pinky[1] + 0.5 * pinky_tip_coords[1])
                        )


                    else:
                        smooth_thumb = thumb_tip_coords
                        smooth_index = index_tip_coords
                        smooth_middle = middle_tip_coords
                        smooth_ring = ring_tip_coords
                        smooth_pinky = pinky_tip_coords

                    
                    right_thumb = smooth_thumb 
                    right_index = smooth_index 
                    right_middle = smooth_middle 
                    right_ring = smooth_ring 
                    right_pinky = smooth_pinky 

                    prev_right_thumb = smooth_thumb 
                    prev_right_index = smooth_index
                    prev_right_middle = smooth_middle
                    prev_right_ring = smooth_ring 
                    prev_right_pinky = smooth_pinky


                elif hand_label == "Left":

                    left_pinch = distance_index_thumb < pinch_threshold 
                    left_middle_pinch = distance_middle_thumb < pinch_threshold 
                    left_ring_pinch = distance_ring_thumb < pinch_threshold 
                    left_pinky_pinch = distance_pinky_thumb < pinch_threshold 


                    if prev_left_thumb is not None and prev_left_index is not None and prev_left_middle is not None and prev_left_ring is not None and prev_left_pinky is not None:
                        
                        smooth_thumb = (
                            int(0.5 * prev_left_thumb[0] + 0.5 * thumb_tip_coords[0]),
                            int(0.5 * prev_left_thumb[1] + 0.5 * thumb_tip_coords[1])
                        )

                        smooth_index = (
                            int(0.5 * prev_left_index[0] + 0.5 * index_tip_coords[0]),
                            int(0.5 * prev_left_index[1] + 0.5 * index_tip_coords[1])
                        )

                        smooth_middle = (
                            int(0.5 * prev_left_middle[0] + 0.5 * middle_tip_coords[0]),
                            int(0.5 * prev_left_middle[1] + 0.5 * middle_tip_coords[1])
                        )

                        smooth_ring = (
                            int(0.5 * prev_left_ring[0] + 0.5 * ring_tip_coords[0]),
                            int(0.5 * prev_left_ring[1] + 0.5 * ring_tip_coords[1])
                        )

                        smooth_pinky = (
                            int(0.5 * prev_left_pinky[0] + 0.5 * pinky_tip_coords[0]),
                            int(0.5 * prev_left_pinky[1] + 0.5 * pinky_tip_coords[1])
                        )


                    else:
                        smooth_thumb = thumb_tip_coords
                        smooth_index = index_tip_coords
                        smooth_middle = middle_tip_coords
                        smooth_ring = ring_tip_coords
                        smooth_pinky = pinky_tip_coords

                    
                    left_thumb = smooth_thumb 
                    left_index = smooth_index 
                    left_middle = smooth_middle 
                    left_ring = smooth_ring 
                    left_pinky = smooth_pinky 

                    prev_left_thumb = smooth_thumb 
                    prev_left_index = smooth_index
                    prev_left_middle = smooth_middle
                    prev_left_ring = smooth_ring 
                    prev_left_pinky = smooth_pinky 


                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        
        # all_pinched = right_pinch and right_middle_pinch and right_ring_pinch and right_pinky_pinch and left_pinch and left_middle_pinch and left_ring_pinch and left_pinky_pinch
        # if all_pinched and not prev_all_pinched:
        #     portal_activate = not portal_activate

        # prev_all_pinched = all_pinched


        both_pinched = right_pinch and left_pinch
        if both_pinched and not prev_both_pinched:
            portal_activate = not portal_activate

        prev_both_pinched = both_pinched




        if (
            all([left_index, left_thumb, right_thumb, right_index]) and 
            portal_activate):

            points = np.array([left_index, left_thumb, right_thumb, right_index], np.int32)

            cv.polylines(frame, [points], isClosed = True, color = (0, 255, 0), thickness = 3)

            mask_frame = np.zeros((h, w), dtype = np.uint8)

            cv.fillPoly(mask_frame, [points], color = 255)



        # --------------- watercolour -------------
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # effect_frame = cv.stylization(
            #     gray,
            #     sigma_s = 60,
            #     sigma_r = 0.45
            # )


        # --------------- thermal -----------------
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # effect_frame = cv.applyColorMap(
            #     gray,
            #     cv.COLORMAP_JET
            # )


        #  -------- edge detection -----------
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # edges = cv.Canny(gray, 80, 150)

            # effect_frame = cv.cvtColor(edges,
            #                     cv.COLOR_GRAY2BGR)


        # ----------- stippling -----------------------
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # effect_frame = np.full_like(frame, 255)

            # for y in range(0, h, 4):
            #     for x in range(0, w, 4):

            #         intensity = gray[y,x]

            #         radius = int((255 - intensity)/80)

            #         if radius > 0:
            #             cv.circle(effect_frame, (x,y), radius, (0,0,0), -1)

        # --------------- pencil sketch --------------------
            # gray_sketch, color_sketch = cv.pencilSketch(
            #     frame,
            #     sigma_s=60,
            #     sigma_r=0.07,
            #     shade_factor=0.05
            # )

            # effect = cv.cvtColor(gray_sketch,
            #                     cv.COLOR_GRAY2BGR)

        # --------- cartoon ---------------------
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # gray = cv.medianBlur(gray, 5)

            # edges = cv.adaptiveThreshold(
            #     gray,
            #     255,
            #     cv.ADAPTIVE_THRESH_MEAN_C,
            #     cv.THRESH_BINARY,
            #     9,
            #     9
            # )

            # color = cv.bilateralFilter(frame,
            #                         9,
            #                         300,
            #                         300)

            # effect = cv.bitwise_and(
            #     color,
            #     color,
            #     mask=edges
            # )

        # --------- pixelization ---------------------
            # small = cv.resize(
            #     frame,
            #     (w//10, h//10),
            #     interpolation=cv.INTER_LINEAR
            # )

            # effect_frame = cv.resize(
            #     small,
            #     (w,h),
            #     interpolation=cv.INTER_NEAREST
            # )

        # --------- risograph --------------------
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            gray = (gray//64)*64

            effect_frame = np.zeros_like(frame)

            effect_frame[:,:,0] = gray * 0.9
            effect_frame[:,:,1] = gray * 0.2
            effect_frame[:,:,2] = gray

            noise = np.random.randint(
                0,
                30,
                frame.shape,
                dtype=np.uint8
            )

            effect_frame = cv.add(effect_frame, noise)





            frame = np.where(
                mask_frame[:, :, None] == 255,
                effect_frame,
                frame
            )


        cv.imshow("..", frame)

        k = cv.waitKey(1)

        if k == 27:
            break


    cap.release()
    cv.destroyAllWindows()

multi_feature()




