import cv2
import time

from numpy import size

class cap:
    def __init__(self, port=1):
        # Initialize the camera (0 is usually the default camera)
        self.vc = cv2.VideoCapture(port)
        if not self.vc.isOpened():
            print("Error: Could not open camera")
            exit()
        self.time = time.time()
        self.capture()
        self.get_size()
    
    def get_size(self):
        """Get current video capture resolution"""
        width = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (width, height)
        return (width, height)
    
    def set_size(self, width, height):
        """Set video capture resolution"""
        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_size = self.get_size()
        print(f"Requested: {width}x{height}, Actual: {actual_size[0]}x{actual_size[1]}")
        return actual_size
    
    def capture(self):
        # Capture a single frame
        ret, frame = self.vc.read()
        if ret:
            # Save the captured frame as an image
            filename = f'captured_photo_{int(self.time)}.jpg'
            cv2.imwrite(filename, frame)
        else:
            print("Error: Could not capture photo")
    
    def __len__(self):
        return int(self.size[0] * self.size[1] * 3)  # Assuming 3 channels (BGR)


# Example usage:
# Default resolution
#a = cap()
#print("Camera capture complete.")
#print(size(a))

# Or specify custom resolution (e.g., 1920x1080, 1280x720, 640x480)
# b = cap(port=1, width=1280, height=720)