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
    cap = cv.VideoCapture(1)

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

        elapsed = time.time() - start
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    cv.destroyAllWindows()



def main():
    print("Hello World")
    run_cam()

if __name__ == "__main__":
    main()