### work on acuracy for mid-point noting on flagged [bounded] rectangles ###
### work on active condensing for multiple flags around shadows for the same marker ###
from test import reconstruct_from_measured_points, plot_measured_and_estimated_curve, plot_xz_points_with_parabola
from testshape import build_reference_parabola
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# good val for static reduction while catching 2-3m distance

minimum_flagged_area = 100

# ranges for yellow detection

upper_bound_yellow = np.array([30, 255, 255])  # don't change
lower_bound_yellow = np.array([20, 100, 100])  # alter as needed : currently flags glazed chair wood (darker orangish) ~hex aa632b 965511 a5632f
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
    def measure(self,measurements):
            self.m = measurements
    def trip(self):
                if self.m['flags'] > self.evalues['flags']:
                    return self.etypes["flags"]
                if measurements['sag'] > self.evalues['sag']:
                    return self.etypes["sag"]
                if measurements['temp'] > self.evalues['temp']:
                    return self.etypes["temp"]
                if measurements["wind"] > self.evalues['wind']:
                    return self.etypes["wind"]
    
    
    

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

def main():
    print("Starting AEPrototype..")
    run_cam(live=True, save_compare_png=False)

if __name__ == "__main__":
    main()