### work on acuracy for mid-point noting on flagged [bounded] rectangles ###
### work on active condensing for multiple flags around shadows for the same marker ###
from test import reconstruct_from_measured_points, plot_measured_and_estimated_curve, plot_xz_points_with_parabola
from testshape import build_reference_parabola
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tkinter as tk
from datetime import datetime
import time
import json
import urllib.error
import urllib.request
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

# good val for static reduction while catching 2-3m distance

minimum_flagged_area = 100

# ranges for yellow detection

upper_bound_yellow = np.array([30, 255, 255])  # don't change
lower_bound_yellow = np.array([20, 100, 100])  # alter as needed : currently flags glazed chair wood (darker orangish) ~hex aa632b 965511 a5632f


def fetch_columbus_weather():
    """
    Fetch current weather for Columbus, OH from weather.gov nearest observation station.

    Returns:
        dict with keys: temp_c, wind_mph, station_id
    Raises:
        RuntimeError if API data cannot be retrieved/parsed.
    """
    lat, lon = 39.9612, -82.9988
    headers = {
        "User-Agent": "Make26SagTracker/1.0 (local app)",
        "Accept": "application/geo+json",
    }

    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    req_points = urllib.request.Request(points_url, headers=headers)
    with urllib.request.urlopen(req_points, timeout=6) as resp:
        points_data = json.loads(resp.read().decode("utf-8"))

    stations_url = points_data.get("properties", {}).get("observationStations")
    if not stations_url:
        raise RuntimeError("No observation stations URL returned by weather.gov points API")

    req_stations = urllib.request.Request(stations_url, headers=headers)
    with urllib.request.urlopen(req_stations, timeout=6) as resp:
        stations_data = json.loads(resp.read().decode("utf-8"))

    features = stations_data.get("features", [])
    if not features:
        raise RuntimeError("No nearby weather stations found for Columbus")

    station_id = features[0].get("properties", {}).get("stationIdentifier")
    if not station_id:
        raise RuntimeError("Station identifier missing from weather.gov response")

    latest_url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
    req_latest = urllib.request.Request(latest_url, headers=headers)
    with urllib.request.urlopen(req_latest, timeout=6) as resp:
        latest_data = json.loads(resp.read().decode("utf-8"))

    props = latest_data.get("properties", {})
    temp_c = props.get("temperature", {}).get("value")
    wind_mps = props.get("windSpeed", {}).get("value")

    if temp_c is None or wind_mps is None:
        raise RuntimeError("Missing temperature or wind speed in latest observation")

    wind_mph = float(wind_mps) * 2.2369362921
    return {
        "temp_c": float(temp_c),
        "wind_mph": float(wind_mph),
        "station_id": str(station_id),
    }
class faultresponse():

    def __init__(self, values = {'flags': 10, 'sag': .05, 
                                'temp':50 , 'm': None, "wind":40}):
                    self.etypes = {
                                'flags': "cant detect flags requires maintenance", 'sag': "sag too much in danger", 
                                'temp': "temperature too high", 'm': "maintenance required", "wind": "WIND TOO HIGH"
                    }
                    self.evalues =  {
                                'flags': values["flags"], 'sag': values["sag"], 
                                'temp':values["temp"] , 'm': None, "wind":values["wind"]
                    } 
    def measure(self, measurements):
            self.m = measurements
    def trip(self):
                if self.m is None:
                    return None
                if self.m['flags'] < self.evalues['flags']:
                    return self.etypes["flags"]
                if self.m['sag'] > self.evalues['sag']:
                    return self.etypes["sag"]
                if self.m['temp'] > self.evalues['temp']:
                    return self.etypes["temp"]
                if self.m["wind"] > self.evalues['wind']:
                    return self.etypes["wind"]
                return None
    
    
    

def save_original_vs_estimated_png(
    measured_points,
    original_sag_m,
    span_m=100.0,
    tilt_deg=30.0,
    camera_height_m=1.0,
    origin=(120.0, 260.0),
    pixels_per_meter=None,
    output_png="benchmark_outputs/original_vs_estimated_parabola.png",
):
    """Save an X-Z comparison PNG of original parabola vs estimated parabola from measured points."""
    if len(measured_points) < 3:
        raise ValueError("Need at least 3 measured points to compare parabolas.")

    estimate = reconstruct_from_measured_points(
        measured_points,
        span=span_m,
        tilt_deg=tilt_deg,
        camera_height_m=camera_height_m,
        origin=origin,
        pixels_per_meter=pixels_per_meter,
    )

    recon = np.array(estimate["reconstructed_points"], dtype=float)
    rx = recon[:, 0]
    rz = recon[:, 2]

    fit = estimate["fitted_parabola"]
    a, b, c = float(fit["a"]), float(fit["b"]), float(fit["c"])
    x_dense = np.linspace(float(rx.min()), float(rx.max()), 300)
    z_est = a * x_dense * x_dense + b * x_dense + c

    ref = np.array(build_reference_parabola(span=span_m, sag=original_sag_m, num_points=len(measured_points)), dtype=float)
    ref_x = ref[:, 0]
    ref_z = ref[:, 2]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(ref_x, ref_z, color="tab:green", linewidth=2.0, label=f"Original parabola (sag={original_sag_m:.2f} m)")
    ax.scatter(rx, rz, s=24, color="tab:blue", alpha=0.85, label="Reconstructed points")
    ax.plot(x_dense, z_est, color="tab:orange", linewidth=2.0, label="Estimated parabola")
    ax.set_title("Original vs Estimated Parabola (X-Z)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


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


def run_cam(
    live=False,
    save_compare_png=False,
    original_sag_m=2.0,
    compare_png_path="benchmark_outputs/original_vs_estimated_parabola.png",
):

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

    saved_once = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Error: Failed to grab frame")
            break

        processed_frame, mask, points = flag_detection(frame)
        if live == True and len(points) >= 3:
            frompoints = reconstruct_from_measured_points(points, span=100, camera_height_m=1)
            a = plot_xz_points_with_parabola(frompoints)######this is the live plotting you need
            

        if save_compare_png and (not saved_once) and len(points) >= 3:
            out = save_original_vs_estimated_png(
                points,
                original_sag_m=original_sag_m,
                span_m=100.0,
                camera_height_m=1.0,
                output_png=compare_png_path,
            )
            print(f"Saved comparison PNG: {out}")
            saved_once = True
        
        if live == True:
            e = faultresponse()
            e.measure({
                                'flags': 30, 'sag': 5, 
                                'temp': 30, "wind": 20
                    })
            e.trip()
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


class SagTrackerUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AEP Line Sag Tracker")
        self.root.geometry("1000x700")

        # Stored camera capture handle for the active camera module screen.
        self.cap = None
        self.current_frame = None
        self.latest_points = []
        self.photo = None
        self.frame_count = 0
        self.latest_weather = None
        self.module_weather = {}
        self.last_weather_refresh_ts = 0.0

        self.main_menu = tk.Frame(self.root)
        self.camera_screen = tk.Frame(self.root)
        self.register_screen = tk.Frame(self.root)
        self.bug_report_screen = tk.Frame(self.root)
        self.registered_camera_modules = []
        self.bug_reports = []

        self._build_main_menu()
        self._build_camera_screen()
        self._build_register_screen()
        self._build_bug_report_screen()
        self.show_main_menu()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_main_menu(self):
        title = tk.Label(self.main_menu, text="Main Menu", font=("Segoe UI", 20, "bold"))
        title.pack(pady=(20, 8))

        subtitle = tk.Label(self.main_menu, text="Connected Modules", font=("Segoe UI", 13))
        subtitle.pack(pady=(0, 10))

        self.main_modules_list = tk.Listbox(self.main_menu, height=7, width=56, font=("Segoe UI", 11))
        self.main_modules_list.pack(pady=(0, 16))

        open_camera_btn = tk.Button(
            self.main_menu,
            text="Open Camera Module",
            font=("Segoe UI", 12, "bold"),
            padx=14,
            pady=8,
            command=self.show_camera_screen,
        )
        open_camera_btn.pack()

        register_btn = tk.Button(
            self.main_menu,
            text="Register New Camera Module",
            font=("Segoe UI", 11),
            padx=12,
            pady=6,
            command=self.show_register_screen,
        )
        register_btn.pack(pady=(8, 0))

        bug_btn = tk.Button(
            self.main_menu,
            text="Open Bug Reports",
            font=("Segoe UI", 11),
            padx=12,
            pady=6,
            command=self.show_bug_report_screen,
        )
        bug_btn.pack(pady=(8, 0))

    def _build_camera_screen(self):
        self.camera_screen.grid_rowconfigure(0, weight=3)
        self.camera_screen.grid_rowconfigure(1, weight=2)
        self.camera_screen.grid_columnconfigure(0, weight=3)
        self.camera_screen.grid_columnconfigure(1, weight=2)

        top_left = tk.LabelFrame(self.camera_screen, text="Live Camera", padx=8, pady=8)
        top_left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        top_right = tk.LabelFrame(self.camera_screen, text="Controls / Module Switch", padx=8, pady=8)
        top_right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        bottom_left = tk.LabelFrame(self.camera_screen, text="X-Z Plot", padx=8, pady=8)
        bottom_left.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        bottom_right = tk.LabelFrame(self.camera_screen, text="Print / Errors", padx=8, pady=8)
        bottom_right.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)

        self.video_label = tk.Label(top_left, text="Camera feed appears here", bg="#111", fg="white")
        self.video_label.pack(fill="both", expand=True)

        self.status_label = tk.Label(top_right, text="Camera idle", font=("Segoe UI", 11), anchor="w")
        self.status_label.pack(fill="x", pady=(0, 8))

        module_subtitle = tk.Label(top_right, text="Connected Modules", font=("Segoe UI", 10, "bold"), anchor="w")
        module_subtitle.pack(fill="x", pady=(0, 4))

        self.camera_module_list = tk.Listbox(top_right, height=3, font=("Segoe UI", 10))
        self.camera_module_list.selection_set(0)
        self.camera_module_list.pack(fill="x", pady=(0, 6))
        self.camera_module_list.bind("<Double-Button-1>", lambda _evt: self.open_selected_module())

        tk.Button(top_right, text="Open Selected Module", command=self.open_selected_module).pack(fill="x", pady=3)
        tk.Button(top_right, text="Register New Camera Module", command=self.show_register_screen).pack(fill="x", pady=3)
        tk.Button(top_right, text="Open Bug Reports", command=self.show_bug_report_screen).pack(fill="x", pady=3)
        tk.Button(top_right, text="Refresh Module 1 Weather", command=lambda: self.refresh_module_weather(force=True)).pack(fill="x", pady=3)
        tk.Button(top_right, text="Start Camera", command=self.start_camera).pack(fill="x", pady=3)
        tk.Button(top_right, text="Stop Camera", command=self.stop_camera).pack(fill="x", pady=3)
        tk.Button(top_right, text="Save Compare PNG", command=self.save_compare_from_latest).pack(fill="x", pady=3)
        tk.Button(top_right, text="Back to Menu", command=self.show_main_menu).pack(fill="x", pady=3)

        self.weather_module1_label = tk.Label(
            top_right,
            text="Camera Module 1 Weather: loading...",
            justify="left",
            anchor="w",
            bg="#1b1b1b",
            fg="#d7f4ff",
            font=("Consolas", 9),
            padx=8,
            pady=6,
        )
        self.weather_module1_label.pack(fill="x", pady=(6, 4))

        self.error_module_frame = tk.Frame(top_right, bg="#6b0000", highlightthickness=2, highlightbackground="#ff3b30")
        self.error_module_title = tk.Label(
            self.error_module_frame,
            text="⚠️🌋 WARNING CAMERA MODULE 🌋⚠️",
            bg="#6b0000",
            fg="#ffe28a",
            font=("Segoe UI", 11, "bold"),
        )
        self.error_module_title.pack(fill="x", padx=8, pady=(6, 2))

        self.error_module_desc = tk.Label(
            self.error_module_frame,
            text=(
                "Fault visualizer armed.\n"
                "Hazard mode overlays warning status and escalation notes.\n"
                "Use this module to inspect tripped thresholds immediately."
            ),
            justify="left",
            bg="#6b0000",
            fg="#ffd7d1",
            font=("Segoe UI", 9),
        )
        self.error_module_desc.pack(fill="x", padx=8, pady=(0, 8))

        self.plot_figure = Figure(figsize=(5, 3), dpi=100)
        self.plot_ax = self.plot_figure.add_subplot(111)
        self.plot_ax.set_title("Waiting for 3+ points")
        self.plot_ax.set_xlabel("x (m)")
        self.plot_ax.set_ylabel("z (m)")
        self.plot_ax.grid(True, alpha=0.3)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_figure, master=bottom_left)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.plot_canvas.draw()

        self.log_text = tk.Text(bottom_right, height=12, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self.log("Camera module initialized")

        self.refresh_module_lists()

    def _build_register_screen(self):
        title = tk.Label(self.register_screen, text="Register New Camera Module", font=("Segoe UI", 18, "bold"))
        title.pack(pady=(20, 8))

        sub = tk.Label(
            self.register_screen,
            text="Enter camera location information and serial number.",
            font=("Segoe UI", 11),
        )
        sub.pack(pady=(0, 14))

        form = tk.Frame(self.register_screen)
        form.pack(pady=8)

        tk.Label(form, text="Location Info", font=("Segoe UI", 10, "bold"), anchor="w", width=18).grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.location_entry = tk.Entry(form, width=44, font=("Segoe UI", 10))
        self.location_entry.grid(row=0, column=1, padx=6, pady=6)

        tk.Label(form, text="Serial Number", font=("Segoe UI", 10, "bold"), anchor="w", width=18).grid(row=1, column=0, padx=6, pady=6, sticky="w")
        self.serial_entry = tk.Entry(form, width=44, font=("Segoe UI", 10))
        self.serial_entry.grid(row=1, column=1, padx=6, pady=6)

        self.register_status_label = tk.Label(self.register_screen, text="", font=("Segoe UI", 10), fg="#2e7d32")
        self.register_status_label.pack(pady=(8, 8))

        actions = tk.Frame(self.register_screen)
        actions.pack(pady=6)
        tk.Button(actions, text="Save Module", command=self.save_registered_module, padx=12, pady=6).pack(side="left", padx=6)
        tk.Button(actions, text="Back to Main Menu", command=self.show_main_menu, padx=12, pady=6).pack(side="left", padx=6)

        self.register_list = tk.Listbox(self.register_screen, height=8, width=78, font=("Consolas", 10))
        self.register_list.pack(pady=(10, 14))

    def _build_bug_report_screen(self):
        title = tk.Label(self.bug_report_screen, text="Bug Reports", font=("Segoe UI", 18, "bold"))
        title.pack(pady=(20, 8))

        sub = tk.Label(
            self.bug_report_screen,
            text="Submit field bug reports with module metadata and operator details.",
            font=("Segoe UI", 11),
        )
        sub.pack(pady=(0, 12))

        form = tk.Frame(self.bug_report_screen)
        form.pack(pady=6)

        tk.Label(form, text="Location Info", font=("Segoe UI", 10, "bold"), width=16, anchor="w").grid(row=0, column=0, padx=6, pady=5, sticky="w")
        self.bug_location_entry = tk.Entry(form, width=42, font=("Segoe UI", 10))
        self.bug_location_entry.grid(row=0, column=1, padx=6, pady=5)

        tk.Label(form, text="Serial Number", font=("Segoe UI", 10, "bold"), width=16, anchor="w").grid(row=1, column=0, padx=6, pady=5, sticky="w")
        self.bug_serial_entry = tk.Entry(form, width=42, font=("Segoe UI", 10))
        self.bug_serial_entry.grid(row=1, column=1, padx=6, pady=5)

        tk.Label(form, text="Operator", font=("Segoe UI", 10, "bold"), width=16, anchor="w").grid(row=2, column=0, padx=6, pady=5, sticky="w")
        self.bug_operator_entry = tk.Entry(form, width=42, font=("Segoe UI", 10))
        self.bug_operator_entry.grid(row=2, column=1, padx=6, pady=5)

        tk.Label(form, text="Severity", font=("Segoe UI", 10, "bold"), width=16, anchor="w").grid(row=3, column=0, padx=6, pady=5, sticky="w")
        self.bug_severity_var = tk.StringVar(value="Medium")
        severity_menu = tk.OptionMenu(form, self.bug_severity_var, "Low", "Medium", "High", "Critical")
        severity_menu.config(width=39)
        severity_menu.grid(row=3, column=1, padx=6, pady=5, sticky="w")

        tk.Label(form, text="Module", font=("Segoe UI", 10, "bold"), width=16, anchor="w").grid(row=4, column=0, padx=6, pady=5, sticky="w")
        self.bug_module_var = tk.StringVar(value="Camera Capture Module")
        module_menu = tk.OptionMenu(
            form,
            self.bug_module_var,
            "Camera Capture Module",
            "Error Camera Module",
            "Registered Camera Module",
            "UI / Navigation",
            "Other",
        )
        module_menu.config(width=39)
        module_menu.grid(row=4, column=1, padx=6, pady=5, sticky="w")

        tk.Label(form, text="Description", font=("Segoe UI", 10, "bold"), width=16, anchor="nw").grid(row=5, column=0, padx=6, pady=5, sticky="nw")
        self.bug_description_text = tk.Text(form, width=44, height=5, font=("Segoe UI", 10), wrap="word")
        self.bug_description_text.grid(row=5, column=1, padx=6, pady=5)

        self.bug_status_label = tk.Label(self.bug_report_screen, text="", font=("Segoe UI", 10))
        self.bug_status_label.pack(pady=(8, 6))

        controls = tk.Frame(self.bug_report_screen)
        controls.pack(pady=4)
        tk.Button(controls, text="Save Bug Report", command=self.save_bug_report, padx=12, pady=6).pack(side="left", padx=6)
        tk.Button(controls, text="Back to Main Menu", command=self.show_main_menu, padx=12, pady=6).pack(side="left", padx=6)

        self.bug_reports_list = tk.Listbox(self.bug_report_screen, height=10, width=110, font=("Consolas", 9))
        self.bug_reports_list.pack(pady=(10, 14))
        self.refresh_bug_reports_list()

    def refresh_module_lists(self):
        base_main = [
            "Camera Capture Module",
            "Error Camera Module (activates on faults)",
        ]
        base_camera = [
            "Camera Capture Module",
            "Error Camera Module",
        ]

        if hasattr(self, "main_modules_list"):
            self.main_modules_list.delete(0, tk.END)
            for item in base_main:
                self.main_modules_list.insert(tk.END, item)
            for module in self.registered_camera_modules:
                self.main_modules_list.insert(
                    tk.END,
                    f"Registered Camera | {module['location']} | SN {module['serial']}",
                )

        if hasattr(self, "camera_module_list"):
            self.camera_module_list.delete(0, tk.END)
            for item in base_camera:
                self.camera_module_list.insert(tk.END, item)
            for module in self.registered_camera_modules:
                self.camera_module_list.insert(
                    tk.END,
                    f"Registered Camera [{module['serial']}]",
                )
            if self.camera_module_list.size() > 0:
                self.camera_module_list.selection_clear(0, tk.END)
                self.camera_module_list.selection_set(0)

        if hasattr(self, "register_list"):
            self.register_list.delete(0, tk.END)
            if not self.registered_camera_modules:
                self.register_list.insert(tk.END, "No registered camera modules yet.")
            else:
                for idx, module in enumerate(self.registered_camera_modules, start=1):
                    self.register_list.insert(
                        tk.END,
                        f"{idx}. Location: {module['location']} | Serial: {module['serial']}",
                    )

    def refresh_bug_reports_list(self):
        if not hasattr(self, "bug_reports_list"):
            return

        self.bug_reports_list.delete(0, tk.END)
        if not self.bug_reports:
            self.bug_reports_list.insert(tk.END, "No bug reports submitted yet.")
            return

        for idx, r in enumerate(self.bug_reports, start=1):
            self.bug_reports_list.insert(
                tk.END,
                (
                    f"{idx}. [{r['submitted_at']}] [{r['severity']}] [{r['status']}] "
                    f"module={r['module']} | location={r['location']} | serial={r['serial']} | "
                    f"operator={r['operator']} | desc={r['description']}"
                ),
            )

    def show_register_screen(self):
        self.stop_camera()
        self.main_menu.pack_forget()
        self.camera_screen.pack_forget()
        self.bug_report_screen.pack_forget()
        self.register_screen.pack(fill="both", expand=True)
        self.refresh_module_lists()

    def show_bug_report_screen(self):
        self.stop_camera()
        self.main_menu.pack_forget()
        self.camera_screen.pack_forget()
        self.register_screen.pack_forget()
        self.bug_report_screen.pack(fill="both", expand=True)
        self.refresh_bug_reports_list()

    def save_registered_module(self):
        location = self.location_entry.get().strip()
        serial = self.serial_entry.get().strip()

        if not location or not serial:
            self.register_status_label.config(text="Please enter both location info and serial number.", fg="#c62828")
            return

        self.registered_camera_modules.append({"location": location, "serial": serial})
        self.location_entry.delete(0, tk.END)
        self.serial_entry.delete(0, tk.END)
        self.register_status_label.config(text="Camera module registered successfully.", fg="#2e7d32")
        self.refresh_module_lists()
        if hasattr(self, "log_text"):
            self.log(f"Registered camera module | location={location} | serial={serial}")

    def save_bug_report(self):
        location = self.bug_location_entry.get().strip()
        serial = self.bug_serial_entry.get().strip()
        operator = self.bug_operator_entry.get().strip()
        severity = self.bug_severity_var.get().strip()
        module = self.bug_module_var.get().strip()
        description = self.bug_description_text.get("1.0", tk.END).strip()

        if not location or not serial or not operator or not description:
            self.bug_status_label.config(
                text="Please complete location, serial, operator, and description.",
                fg="#c62828",
            )
            return

        report = {
            "location": location,
            "serial": serial,
            "operator": operator,
            "severity": severity,
            "module": module,
            "description": description,
            "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Open",
        }
        self.bug_reports.append(report)
        self.refresh_bug_reports_list()

        self.bug_location_entry.delete(0, tk.END)
        self.bug_serial_entry.delete(0, tk.END)
        self.bug_operator_entry.delete(0, tk.END)
        self.bug_description_text.delete("1.0", tk.END)
        self.bug_severity_var.set("Medium")
        self.bug_module_var.set("Camera Capture Module")

        self.bug_status_label.config(text="Bug report saved.", fg="#2e7d32")
        if hasattr(self, "log_text"):
            self.log(
                f"Bug report saved | severity={severity} | module={module} | serial={serial} | operator={operator}"
            )

    def show_main_menu(self):
        self.stop_camera()
        self.bug_report_screen.pack_forget()
        self.register_screen.pack_forget()
        self.camera_screen.pack_forget()
        self.main_menu.pack(fill="both", expand=True)
        self.refresh_module_lists()

    def show_camera_screen(self):
        self.main_menu.pack_forget()
        self.register_screen.pack_forget()
        self.camera_screen.pack(fill="both", expand=True)
        self.refresh_module_weather(force=True)
        self.start_camera()

    def refresh_module_weather(self, force=False):
        now = time.time()
        if (not force) and (now - self.last_weather_refresh_ts < 60.0):
            return

        try:
            weather = fetch_columbus_weather()
            self.latest_weather = weather
            self.module_weather["Camera Module 1"] = weather
            self.last_weather_refresh_ts = now
            stamp = datetime.now().strftime("%H:%M:%S")
            self.weather_module1_label.config(
                text=(
                    "Camera Module 1 Weather\n"
                    f"Temp: {weather['temp_c']:.1f} C\n"
                    f"Wind: {weather['wind_mph']:.1f} mph\n"
                    f"Station: {weather['station_id']}\n"
                    f"Updated: {stamp}"
                )
            )
        except (RuntimeError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            self.weather_module1_label.config(
                text=(
                    "Camera Module 1 Weather\n"
                    "Temp: unavailable\n"
                    "Wind: unavailable\n"
                    f"Reason: {exc}"
                )
            )
            self.log(f"Module 1 weather refresh failed: {exc}")

    def show_error_module_card(self, reason):
        self.error_module_desc.config(
            text=(
                "⚠️ Fault detected. Escalating to warning camera module.\n"
                f"Trip reason: {reason}\n"
                "Status: RED ALERT / High-visibility monitoring enabled."
            )
        )
        if not self.error_module_frame.winfo_ismapped():
            self.error_module_frame.pack(fill="x", pady=(8, 0))

    def hide_error_module_card(self):
        if self.error_module_frame.winfo_ismapped():
            self.error_module_frame.pack_forget()

    def open_selected_module(self):
        selected = self.camera_module_list.curselection()
        if not selected:
            self.log("No module selected")
            return

        module_name = self.camera_module_list.get(selected[0])
        if module_name == "Camera Capture Module":
            self.log("Camera Capture Module selected")
            self.start_camera()
        elif module_name == "Error Camera Module":
            self.log("Error Camera Module selected")
            self.open_error_module()
        elif module_name.startswith("Registered Camera"):
            self.log(f"{module_name} selected")
            self.status_label.config(text=f"{module_name} selected")
            self.start_camera()

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def start_camera(self):
        if self.cap is not None:
            self.log("Camera already running")
            return

        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.status_label.config(text="Camera error: unable to open camera")
            self.log("Error: unable to open camera at index 1 or 0")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.cap = cap
        self.frame_count = 0
        self.status_label.config(text="Camera running")
        self.log("Camera started")
        self._update_camera_frame()

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.log("Camera stopped")
        self.status_label.config(text="Camera stopped")

    def _update_camera_frame(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_label.config(text="Camera error: failed to grab frame")
            self.log("Error: failed to grab frame")
            self.stop_camera()
            return

        processed_frame, _, points = flag_detection(frame)
        self.latest_points = points
        self.frame_count += 1

        label = f"Flagged Markers: {len(points)}"
        cv2.putText(
            processed_frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            1.1,
            (140, 238, 255),
            2,
        )

        if Image is None or ImageTk is None:
            self.status_label.config(text="Install Pillow for UI video: pip install pillow")
            self.log("Error: Pillow missing")
            self.stop_camera()
            return

        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((960, 540))
        self.photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=self.photo, text="")

        if self.frame_count % 5 == 0:
            self.update_xz_plot(points)
        if self.frame_count % 150 == 0:
            self.refresh_module_weather(force=False)

        self.root.after(20, self._update_camera_frame)

    def update_xz_plot(self, points):
        self.plot_ax.clear()
        self.plot_ax.set_xlabel("x (m)")
        self.plot_ax.set_ylabel("z (m)")
        self.plot_ax.grid(True, alpha=0.3)

        if len(points) < 3:
            self.plot_ax.set_title("Need at least 3 points for reconstruction")
            self.plot_canvas.draw()
            return

        try:
            est = reconstruct_from_measured_points(points, span=100.0, camera_height_m=1.0)
            rec = np.array(est["reconstructed_points"], dtype=float)
            rx = rec[:, 0]
            rz = rec[:, 2]

            fit = est["fitted_parabola"]
            a = float(fit["a"])
            b = float(fit["b"])
            c = float(fit["c"])
            x_dense = np.linspace(float(rx.min()), float(rx.max()), 300)
            z_dense = a * x_dense * x_dense + b * x_dense + c

            self.plot_ax.scatter(rx, rz, s=22, color="tab:blue", label="Reconstructed points")
            self.plot_ax.plot(x_dense, z_dense, color="tab:orange", linewidth=2.0, label="Estimated parabola")
            self.plot_ax.set_title("X-Z Reconstructed Curve")
            self.plot_ax.legend(loc="best")
        except Exception as exc:
            self.plot_ax.set_title("X-Z plot update failed")
            self.log(f"Plot update error: {exc}")

        self.plot_canvas.draw()

    def save_compare_from_latest(self):
        if len(self.latest_points) < 3:
            self.status_label.config(text="Need at least 3 detected points to save compare PNG")
            self.log("Save compare skipped: need at least 3 points")
            return

        out = save_original_vs_estimated_png(
            self.latest_points,
            original_sag_m=2.0,
            span_m=100.0,
            camera_height_m=1.0,
            output_png="benchmark_outputs/original_vs_estimated_parabola.png",
        )
        self.status_label.config(text=f"Saved: {out}")
        self.log(f"Saved comparison PNG: {out}")

    def open_error_module(self):
        sag_est = 0.0
        if len(self.latest_points) >= 3:
            try:
                est = reconstruct_from_measured_points(self.latest_points, span=100.0, camera_height_m=1.0)
                sag_est = float(est.get("estimated_sag_m", 0.0))
            except Exception as exc:
                self.log(f"Error module: failed to estimate sag ({exc})")

        # Pull live local weather for Columbus; fall back if API is unavailable.
        temp_c = 30.0
        wind_mph = 20.0
        weather_note = "(fallback weather values)"
        try:
            weather = fetch_columbus_weather()
            temp_c = float(weather["temp_c"])
            wind_mph = float(weather["wind_mph"])
            self.latest_weather = weather
            weather_note = f"(station {weather['station_id']})"
        except (RuntimeError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            self.log(f"Weather API unavailable, using fallback values: {exc}")

        checker = faultresponse(values={"flags": 3, "sag": 5.0, "temp": 50, "m": None, "wind": 40})
        checker.measure({
            "flags": len(self.latest_points),
            "sag": sag_est,
            "temp": temp_c,
            "wind": wind_mph,
        })
        tripped = checker.trip()

        self.log(
            f"Measurements -> flags={len(self.latest_points)}, sag={sag_est:.2f} m, "
            f"temp={temp_c:.1f} C, wind={wind_mph:.1f} mph {weather_note}"
        )

        if tripped is not None:
            self.log(f"Error tripped: {tripped}")
            self.status_label.config(text=f"Error tripped: {tripped}")
            self.show_error_module_card(tripped)
        else:
            self.log("No errors")
            self.status_label.config(text="No errors")
            self.hide_error_module_card()

    def on_close(self):
        self.stop_camera()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

def main():
    app = SagTrackerUI()
    app.run()

if __name__ == "__main__":
    main()