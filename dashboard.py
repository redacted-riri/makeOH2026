import tkinter as tk
from tkinter import font
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import math
import time

# Import your existing logic!
from main import flag_detection, lower_bound_yellow, upper_bound_yellow, minimum_flagged_area
from weather import cbusWeather  # Updated to match your function name!

class AEPDashboard:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg="#121212")
        self.window.geometry("1400x800")

        # --- Variables ---
        self.current_sag = 0.0
        # Updated keys to match your capitalized dictionary keys
        self.weather_data = {"Temp": "--", "Wind": "--", "Gusts": "--"}
        
        # Try external cam (1), fallback to built-in (0)
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("⚠️ External cam failed, using built-in...")
            self.cap = cv2.VideoCapture(0)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # --- UI Layout ---
        # Left Frame: Camera Feed
        self.left_frame = tk.Frame(window, bg="#121212", bd=0)
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.left_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right Frame: Data & Alerts
        self.right_frame = tk.Frame(window, bg="#1e1e1e", width=400, relief=tk.FLAT)
        self.right_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.Y)
        self.right_frame.pack_propagate(False) # Keep width fixed

        # Title
        title_font = font.Font(family="Helvetica", size=24, weight="bold")
        tk.Label(self.right_frame, text="⚡ AEP Line Monitor", font=title_font, bg="#1e1e1e", fg="#00ffcc").pack(pady=(20, 30))

        # Alert Box (Hidden by default)
        self.alert_var = tk.StringVar()
        self.alert_var.set("✅ STATUS NORMAL")
        self.alert_label = tk.Label(self.right_frame, textvariable=self.alert_var, font=("Helvetica", 16, "bold"), 
                                    bg="#00aa00", fg="white", pady=15)
        self.alert_label.pack(fill=tk.X, padx=20, pady=(0, 30))

        # Sag Angle Display
        tk.Label(self.right_frame, text="Maximum Line Sag", font=("Helvetica", 14), bg="#1e1e1e", fg="#aaaaaa").pack()
        self.sag_var = tk.StringVar()
        self.sag_var.set("--°")
        tk.Label(self.right_frame, textvariable=self.sag_var, font=("Helvetica", 48, "bold"), bg="#1e1e1e", fg="#ffcc00").pack(pady=(0, 30))

        # Weather Display
        tk.Label(self.right_frame, text="Columbus Weather", font=("Helvetica", 14), bg="#1e1e1e", fg="#aaaaaa").pack()
        self.weather_var = tk.StringVar()
        self.weather_var.set("Loading...")
        tk.Label(self.right_frame, textvariable=self.weather_var, font=("Helvetica", 16), bg="#1e1e1e", fg="white", justify=tk.CENTER).pack(pady=10)

        # Manual Refresh Button for Weather
        tk.Button(self.right_frame, text="Refresh Weather", command=self.fetch_weather_thread, 
                  bg="#333333", fg="white", font=("Helvetica", 12), relief=tk.FLAT, padx=10).pack(pady=10)

        # --- Start Loops ---
        self.fetch_weather_thread() # Get initial weather
        self.update_video()         # Start video loop

    def fetch_weather_thread(self):
        threading.Thread(target=self._get_weather, daemon=True).start()

    def _get_weather(self):
        data = cbusWeather() # Updated function call
        if data:
            self.weather_data = data
            # Updated to match your dictionary keys
            self.weather_var.set(f"Temp: {data['Temp']}°F\nWind: {data['Wind']} mph\nGusts: {data['Gusts']} mph")
        else:
            self.weather_var.set("⚠️ API Offline")

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # 1. Run your existing flag detection
            processed_frame, mask, points = flag_detection(frame)
            points.sort(key=lambda p: p[0])

            # 2. Mock calculation for the max sag angle
            # (We will replace this with wire.py math next!)
            if len(points) >= 2:
                max_angle = 0
                for i in range(len(points) - 1):
                    dx = points[i+1][0] - points[i][0]
                    dy = points[i+1][1] - points[i][1]
                    if dx != 0:
                        angle = abs(math.degrees(math.atan2(dy, dx)))
                        if angle > max_angle: max_angle = angle
                self.current_sag = round(max_angle, 1)
            else:
                self.current_sag = 0.0

            # 3. Update the UI Text
            self.sag_var.set(f"{self.current_sag}°")

            # 4. Alert Logic (Using your capitalized 'Wind' key)
            wind_speed = float(self.weather_data.get('Wind', 0) if self.weather_data.get('Wind') != "--" else 0)
            
            if self.current_sag > 15.0 or wind_speed > 20.0:
                self.alert_var.set("⚠️ CRITICAL LINE STRESS")
                self.alert_label.configure(bg="#ff3333", fg="white")
            else:
                self.alert_var.set("✅ STATUS NORMAL")
                self.alert_label.configure(bg="#00aa00", fg="white")

            # 5. Convert OpenCV image (BGR) to Tkinter image (RGB)
            cv2_im = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_im)
            img = img.resize((960, 540), Image.Resampling.LANCZOS) 
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk 
            self.video_label.configure(image=imgtk)

        self.window.after(15, self.update_video)

    def on_closing(self):
        print("Closing application...")
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AEPDashboard(root, "AEP Line Stress Monitor")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()