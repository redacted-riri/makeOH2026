### work on acuracy for mid-point noting on flagged [bounded] rectangles ###
### work on active condensing for multiple flags around shadows for the same marker ###

import cv2
import numpy as np

# good val for static reduction while catching 2-3m distance

minimum_flagged_area = 100

# ranges for yellow detection

upper_bound_yellow = np.array([30, 255, 255])  # don't change
lower_bound_yellow = np.array([20, 100, 100])  # alter as needed : currently flags glazed chair wood (darker orangish) ~hex aa632b 965511 a5632f


def flag_detection(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound_yellow, upper_bound_yellow)
    kernel = np.ones((3,3), np.uint8)


    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # remov minimal static / false pos

    mask = cv2.dilate(mask, kernel, iterations=1) # thicken markers : fill gaps : join tears 



    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    flag_cords = [] 
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > minimum_flagged_area:

            x, y, w, h = cv2.boundingRect(contour)

            centr = (x + w // 2, y + h // 2) # center @ midpoint (x,y) of l & w of bounding box

            flag_cords.append(centr)
            
            # draw bounding box / center dot
            cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 255, 57), 2)
            cv2.circle(frame, centr, 4, (63, 0, 255), -1) 

    return frame, mask, flag_cords


def run_cam():

    # try to grab the external webcam feed (i=0)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("⚠️ Error: Could not open external webcam (index 1) - Trying main camera (index 0)") # will also occur if xtrnal cam is in use
        cap = cv2.VideoCapture(0) # fallback to default cam

    # force proper resolution render, may need to rm 64/65 on pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  

    print("✅ Camera Active - Tracking Yellow Markers")
    print("(to quit type 'x' with the feedback window selected)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Error: Failed to grab frame")
            break

        processed_frame, mask, points = flag_detection(frame)
        
        # sort points logged by X values (L -> R)
        points.sort(key=lambda p: p[0])

        # draw label on view window
        text = f"Flagged Markers: {len(points)}"
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.2
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        
        cv2.putText(
            processed_frame,
            text,
            (20, 25 + text_h//2),
            font,
            scale,
            (140, 238, 255),
            thickness
        )
        
        # display window headers
        cv2.imshow("AEP Line Sag Tracker", processed_frame)
        cv2.imshow("Detection Key | White = Yellow", mask)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Starting AEPrototype..")
    run_cam()

if __name__ == "__main__":
    main()