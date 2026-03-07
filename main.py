from cam import *
#from temp import *
import numpy as np
import cv2 as cv
from datetime import datetime

def find_black(frame):
    #im = cv.imread('test4.png')
    #assert im is not None, "file could not be read"

    imgray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # detect dark pixels (0–50)
    mask = cv.inRange(imgray, 0, 50)
    black_pixels = cv.countNonZero(mask)

    return black_pixels

def run_cam():
    cap = cv.VideoCapture(0)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not cap.isOpened():
        print("Could not open camera")
        return

    target_fps = 15
    frame_time = 1 / target_fps

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        black_count = find_black(frame)
        cv.putText(frame, f"Black: {black_count}", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv.putText(frame, f"Time: {timestamp}", (20,80), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        print(f"{timestamp} | Black pixels: {black_count}")

        cv.imshow("Camera", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        detect_flags()
        elapsed = time.time() - start
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    cv.destroyAllWindows()

def detect_flags():
    

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Yellow = (0, 255, 255) in BGR
            cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)

        cv2.imshow('Contour Approximation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("Hello World")
    run_cam()

if __name__ == "__main__":
    main()