import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

# Magnification strength
ZOOM = 2.0

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    if not ret:
        break

    h, w = frame.shape[:2]

    # Example polygon (hexagon-ish)
    polygon = np.array([
        [250, 150],
        [400, 150],
        [450, 250],
        [400, 350],
        [250, 350],
        [200, 250]
    ], dtype=np.int32)

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # Bounding rectangle around polygon
    x, y, pw, ph = cv2.boundingRect(polygon)

    # Crop original region
    crop = frame[y:y+ph, x:x+pw]

    # Magnify region
    magnified = cv2.resize(
        crop,
        None,
        fx=ZOOM,
        fy=ZOOM,
        interpolation=cv2.INTER_LINEAR
    )

    # Center crop back to original polygon size
    mh, mw = magnified.shape[:2]

    cx = (mw - pw) // 2
    cy = (mh - ph) // 2

    magnified_crop = magnified[
        cy:cy+ph,
        cx:cx+pw
    ]

    # Create polygon mask for local ROI
    local_mask = np.zeros((ph, pw), dtype=np.uint8)

    # Shift polygon coordinates to local ROI space
    local_polygon = polygon - np.array([x, y])

    cv2.fillPoly(local_mask, [local_polygon], 255)

    # Extract polygon area from magnified crop
    magnified_poly = cv2.bitwise_and(
        magnified_crop,
        magnified_crop,
        mask=local_mask
    )

    # Extract inverse area from original frame
    inverse_mask = cv2.bitwise_not(local_mask)

    original_roi = frame[y:y+ph, x:x+pw]

    background = cv2.bitwise_and(
        original_roi,
        original_roi,
        mask=inverse_mask
    )

    # Combine both
    final_roi = cv2.add(background, magnified_poly)

    # Put back into frame
    frame[y:y+ph, x:x+pw] = final_roi

    # Draw polygon outline
    cv2.polylines(
        frame,
        [polygon],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2
    )

    cv2.imshow("Polygon Magnifier", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()