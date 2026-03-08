import cv2
import numpy as np
import math
import numbers

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def calculate_vector(p1, p2):
    """Calculate vector from p1 to p2 for 2D/3D tuples."""
    if len(p1) != len(p2):
        raise ValueError("Points must have the same dimension")
    return tuple(p2[i] - p1[i] for i in range(len(p1)))

def calculate_length(vector):
    """Calculate vector magnitude for any dimension."""
    return math.sqrt(sum(component ** 2 for component in vector))

def calculate_angle(vector):
    """Calculate angle of vector in degrees (from positive x-axis)"""
    return math.degrees(math.atan2(vector[1], vector[0]))

def calculate_angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in degrees"""
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    len1 = calculate_length(v1)
    len2 = calculate_length(v2)
    if len1 == 0 or len2 == 0:
        return 0
    cos_angle = dot_product / (len1 * len2)
    # Clamp to avoid numerical errors
    cos_angle = max(-1, min(1, cos_angle))
    return math.degrees(math.acos(cos_angle))


def calculate_vector_3d(p1, p2):
    """Calculate a 3D vector from p1 to p2."""
    return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])


def calculate_length_3d(vector):
    """Calculate 3D vector magnitude."""
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def display_vectors_3d(points_3d, show_labels=True, show_points=True, title="3D Vectors", show_2d_projection=False):
    """
    Display vectors between consecutive 3D points using matplotlib.

    Args:
        points_3d: List of (x, y, z) tuples.
        show_labels: Draw length labels at segment midpoints.
        show_points: Draw point markers and point labels.
        title: Plot title.
        show_2d_projection: If True, shows 2D X-Y projection to the right of 3D view.

    Returns:
        List of dicts with vector data: from, to, vector, length.
    """
    if plt is None:
        raise ImportError("matplotlib is required for 3D display. Install with: pip install matplotlib")

    if len(points_3d) < 2:
        raise ValueError("Need at least 2 points to display vectors in 3D.")

    # Create figure with subplots if 2D projection is requested
    if show_2d_projection:
        fig = plt.figure(figsize=(16, 7))
        ax = fig.add_subplot(121, projection="3d")
    else:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    vectors_info = []
    xs = [p[0] for p in points_3d]
    ys = [p[1] for p in points_3d]
    zs = [p[2] for p in points_3d]

    if show_points:
        ax.scatter(xs, ys, zs, c="tab:blue", s=35, label="Points")
        for i, p in enumerate(points_3d):
            ax.text(p[0], p[1], p[2], f"P{i}", fontsize=8)

    # Collect vector endpoints for red dots
    vector_ends_x = []
    vector_ends_y = []
    vector_ends_z = []

    for i in range(len(points_3d) - 1):
        p1 = points_3d[i]
        p2 = points_3d[i + 1]
        v = calculate_vector_3d(p1, p2)
        length = calculate_length_3d(v)

        vectors_info.append({
            "from": p1,
            "to": p2,
            "vector": v,
            "length": length,
        })

        ax.quiver(
            p1[0], p1[1], p1[2],
            v[0], v[1], v[2],
            arrow_length_ratio=0.1,
            color="tab:green",
            linewidth=2,
        )

        # Add vector endpoint for red dot
        vector_ends_x.append(p2[0])
        vector_ends_y.append(p2[1])
        vector_ends_z.append(p2[2])

        if show_labels:
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)
            ax.text(mid[0], mid[1], mid[2], f"L={length:.2f}", color="tab:red", fontsize=8)

    # Draw red dots at vector endpoints
    if vector_ends_x:
        ax.scatter(vector_ends_x, vector_ends_y, vector_ends_z, c="red", s=50, label="Vector ends", marker='o')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    ax.legend(loc="best")

    # Add 2D projection if requested
    if show_2d_projection:
        ax2 = fig.add_subplot(122)
        _draw_2d_projection(ax2, points_3d, vectors_info, show_labels, show_points)

    plt.tight_layout()
    plt.show()

    return vectors_info


def _draw_2d_projection(ax, points_3d, vectors_info, show_labels, show_points):
    """
    Draw 2D X-Z projection of 3D vectors.
    
    Args:
        ax: Matplotlib axis to draw on.
        points_3d: List of (x, y, z) tuples.
        vectors_info: List of vector information dicts.
        show_labels: Whether to show length labels.
        show_points: Whether to show point markers.
    """
    xs = [p[0] for p in points_3d]
    zs = [p[2] for p in points_3d]

    if show_points:
        ax.scatter(xs, zs, c="tab:blue", s=35, label="Points", zorder=3)
        for i, p in enumerate(points_3d):
            ax.text(p[0], p[2], f"P{i}", fontsize=8, ha='right')

    # Draw vectors
    vector_ends_x = []
    vector_ends_z = []
    
    for i, info in enumerate(vectors_info):
        p1 = info['from']
        p2 = info['to']
        
        # Draw arrow
        ax.annotate('', xy=(p2[0], p2[2]), xytext=(p1[0], p1[2]),
                   arrowprops=dict(arrowstyle='->', color='tab:green', lw=2))
        
        vector_ends_x.append(p2[0])
        vector_ends_z.append(p2[2])
        
        if show_labels:
            mid_x = (p1[0] + p2[0]) / 2
            mid_z = (p1[2] + p2[2]) / 2
            ax.text(mid_x, mid_z, f"L={info['length']:.2f}", 
                   color="tab:red", fontsize=8, ha='center')
    
    # Draw red dots at vector endpoints
    if vector_ends_x:
        ax.scatter(vector_ends_x, vector_ends_z, c="red", s=50, 
                  label="Vector ends", marker='o', zorder=4)
    
    ax.set_title("2D Projection (X-Z)")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_aspect('equal', adjustable='datalim')


def angle_between_vectors_3d(v1, v2):
    """Calculate angle between two 3D vectors in degrees."""
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    len1 = calculate_length_3d(v1)
    len2 = calculate_length_3d(v2)
    if len1 == 0 or len2 == 0:
        return 0.0
    cos_theta = dot / (len1 * len2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))

def draw_vector(image, p1, p2, color=(0, 255, 0), thickness=2, arrow=True):
    """Draw a vector from p1 to p2 on the image"""
    p1_int = (int(p1[0]), int(p1[1]))
    p2_int = (int(p2[0]), int(p2[1]))
    
    if arrow:
        cv2.arrowedLine(image, p1_int, p2_int, color, thickness, tipLength=0.3)
    else:
        cv2.line(image, p1_int, p2_int, color, thickness)
    
    return image

def draw_vectors_between_points(image, points, color=(0, 255, 0), thickness=2, 
                                show_info=True, arrow=True):
    """
    Draw vectors between consecutive points and display length and angle info
    
    Args:
        image: Image to draw on
        points: List of (x, y) tuples
        color: Color of the vectors (B, G, R)
        thickness: Line thickness
        show_info: Whether to display length and angle text
        arrow: Whether to draw arrows or simple lines
    
    Returns:
        image: Modified image with vectors drawn
        vectors_info: List of dictionaries with vector information
    """
    vectors_info = []
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        
        # Calculate vector properties
        vector = calculate_vector(p1, p2)
        length = calculate_length(vector)
        angle = calculate_angle(vector)
        
        # Store info
        info = {
            'from': p1,
            'to': p2,
            'vector': vector,
            'length': length,
            'angle': angle
        }
        vectors_info.append(info)
        
        # Draw the vector
        draw_vector(image, p1, p2, color, thickness, arrow)
        
        # Draw info text if requested
        if show_info:
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)
            text = f"L:{length:.1f} A:{angle:.1f}°"
            cv2.putText(image, text, (mid_x, mid_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw points
        cv2.circle(image, (int(p1[0]), int(p1[1])), 5, (255, 0, 0), -1)
    
    # Draw last point
    if points:
        last_point = points[-1]
        cv2.circle(image, (int(last_point[0]), int(last_point[1])), 5, (255, 0, 0), -1)
    
    return image, vectors_info
class vector:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.vector = calculate_vector(p1, p2)
        self.lengthp = calculate_length(self.vector)#needs to be modified to be in real world units
        self.correctionfactor = 1 #this can be modified based on calibration to convert pixel length to real world units
        self.length = self.lengthp * self.correctionfactor
        self.angle = calculate_angle(self.vector)
        self.chlen = 1 #meters
    def __add__(self, other):
        if isinstance(other, vector):
            if np.size(self.vector) != np.size(other.vector):
                raise ValueError("Vectors must be of the same dimension to add")
            else:
                new_vector = tuple(self.vector[i] + other.vector[i] for i in range(len(self.vector)))
                origin = tuple(0 for _ in range(len(new_vector)))
                return vector(origin, new_vector)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            new_vector = tuple(self.vector[i] * other for i in range(len(self.vector)))
            origin = tuple(0 for _ in range(len(new_vector)))
            return vector(origin, new_vector)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)
    def mag(self):
        a = 0
        for i in range(len(self.vector)):
            a += self.vector[i]**2
        return math.sqrt(a)
    import numpy as np

    def convertspace(self, newspace, type1 = "cartesian", type2 = "cartesian"):
        if type(newspace) != vector:
            raise ValueError("New space must be a vector")
        if np.size(self.vector) != 3:
            raise ValueError("please use size 3")
        if np.size(newspace.vector) != 3:
            raise ValueError("please use size 3")
        if type1 == "cylindrical":
            raise NotImplementedError("Cylindrical coordinates not implemented yet")
        if type1 == "spherical":
            raise NotImplementedError("Spherical coordinates not implemented yet")
        if type2 == "cylindrical":
            raise NotImplementedError("Cylindrical coordinates not implemented yet")
        if type2 == "spherical":
            raise NotImplementedError("Spherical coordinates not implemented yet")
        if type1 == "cartesian" and type2 == "cartesian":
                """Get DCM that rotates vector v to vector vx."""
                v = np.array(self.vector, dtype=float)
                vx = np.array(newspace.vector, dtype=float)

                nv = np.linalg.norm(v)
                nvx = np.linalg.norm(vx)
                if nv == 0 or nvx == 0:
                    raise ValueError("Cannot build DCM from zero-length vector")

                v = v / nv
                vx = vx / nvx
                dot = float(np.clip(np.dot(v, vx), -1.0, 1.0))

                if np.isclose(dot, 1.0):
                    self.dcm = np.eye(3)
                    return self.dcm

                if np.isclose(dot, -1.0):
                    # 180-degree case: choose any stable orthogonal axis.
                    helper = np.array([1.0, 0.0, 0.0]) if not np.isclose(abs(v[0]), 1.0) else np.array([0.0, 1.0, 0.0])
                    axis = np.cross(v, helper)
                else:
                    axis = np.cross(v, vx)

                axis_norm = np.linalg.norm(axis)
                if axis_norm == 0:
                    self.dcm = np.eye(3)
                    return self.dcm

                axis = axis / axis_norm
                angle = float(np.arccos(dot))

                # Rodrigues' rotation formula using a unit axis.
                K = np.array([[0.0, -axis[2], axis[1]],
                              [axis[2], 0.0, -axis[0]],
                              [-axis[1], axis[0], 0.0]])

                dcm = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
                self.dcm = dcm
                return self.dcm
    def v2vxbidcm(self):
        if not hasattr(self, "dcm"):
            raise ValueError("DCM is not set. Call convertspace() first.")

        directioncosmat = np.array(self.dcm, dtype=float)
        if directioncosmat.shape != (3, 3):
            raise ValueError("DCM must be a 3x3 matrix")

        v = np.array(self.vector, dtype=float)
        if v.shape != (3,):
            raise ValueError("Vector must be 3D to apply DCM")

        rotated = directioncosmat @ v
        self.vector = (float(rotated[0]), float(rotated[1]), float(rotated[2]))
        return self.vector

    
        

def sort_points(points):
    #sort points by y value hight o low
    return sorted(points, key=lambda p: p[1], reverse=False)


def fit_parabola_open_up(points_3d, min_sag=None):
    """Fit z = ax^2 + bx + c in X-Z plane and force a to be non-negative."""
    if len(points_3d) < 3:
        return points_3d

    xs = np.array([p[0] for p in points_3d], dtype=float)
    ys = np.array([p[1] for p in points_3d], dtype=float)
    zs = np.array([p[2] for p in points_3d], dtype=float)

    x_span = float(np.ptp(xs))
    if x_span < 1e-12:
        return points_3d

    z_span = float(np.ptp(zs))
    if min_sag is None:
        min_sag = max(0.05 * x_span, 0.25)

    if z_span < 1e-9:
        x_mid = float(0.5 * (xs.min() + xs.max()))
        z_end = float(np.mean([zs[0], zs[-1]]))
        sag = float(min_sag)
        a = (4.0 * sag) / (x_span * x_span)
        return [(float(x), float(ys[i]), float(a * (x - x_mid) ** 2 + (z_end - sag))) for i, x in enumerate(xs)]

    a, b, c = np.polyfit(xs, zs, 2)
    if a < 0:
        x_vertex = -b / (2 * a)
        z_vertex = float(np.polyval((a, b, c), x_vertex))
        a = abs(a)
        b = -2.0 * a * x_vertex
        c = a * x_vertex * x_vertex + z_vertex

    return [(float(x), float(ys[i]), float(a * x * x + b * x + c)) for i, x in enumerate(xs)]

def wire_shape(points, fit_equation, chlen = 1/10, cfcl = 1, distance=30, anchor_origin=False, constrain_ends=False, camera_params=None):
    # Image-space assumption: smaller y is farther away, larger y is closer.
    # Process points explicitly from far pole to close pole.
    points = sort_points(points)
    if len(points) < 2:
        return []

    # Build one frame-rotation that maps the observed wire direction to +X.
    dx0 = points[1][0] - points[0][0]
    dy0 = points[1][1] - points[0][1]
    if abs(dx0) < 1e-12 and abs(dy0) < 1e-12:
        source_axis = vector((0, 0, 0), (0, 1, 0))
    else:
        source_axis = vector((0, 0, 0), (dx0, dy0, 0))
    target_axis = vector((0, 0, 0), (1, 0, 0))
    dcm = source_axis.convertspace(target_axis)

    path = [(points[0][0], points[0][1], float(distance))]

    for i in range(len(points) - 1):
        cfcl = characteristic_length(path, len(path) - 1, chlen, distance)
        v = vector((points[i][0], points[i][1], 0), (points[i + 1][0], points[i + 1][1], 0))

        # Convert pixel displacement to meters using a depth-adjusted pixels-per-meter value.
        meters_per_pixel = 1.0 / max(cfcl, 1e-9)
        v = v * meters_per_pixel

        v.dcm = dcm
        step = v.v2vxbidcm()

        prev = path[-1]
        # Keep depth moving from far (distance) toward close pole (0).
        next_z = prev[2] - abs(step[2])
        path.append((prev[0] + step[0], prev[1] + step[1], next_z))

    if fit_equation == "parabola_up":
        # If camera model is provided, invert projection directly.
        if camera_params is not None:
            cam = sort_points(points)
            u = np.array([p[0] for p in cam], dtype=float)
            v = np.array([p[1] for p in cam], dtype=float)

            pixels_per_meter = float(camera_params.get("pixels_per_meter", chlen))
            tilt_deg = float(camera_params.get("tilt_deg", 35.0))
            origin = camera_params.get("origin", (0.0, 0.0))
            camera_height_m = float(camera_params.get("camera_height_m", 0.0))

            tilt = math.radians(tilt_deg)
            height_scale = 1.0 / (1.0 + max(camera_height_m, 0.0))
            k = max(pixels_per_meter * height_scale, 1e-9)

            x_vals = (u - float(origin[0])) / k
            z_vals = (x_vals * math.sin(tilt) - (v - float(origin[1])) / k) / max(math.cos(tilt), 1e-9)
            path = [(float(x_vals[i]), 0.0, float(z_vals[i])) for i in range(len(cam))]
            path = fit_parabola_open_up(path)
        else:
            path = fit_parabola_open_up(path)

    if anchor_origin and path:
        x0, y0, z0 = path[0]
        path = [(x - x0, y - y0, z - z0) for x, y, z in path]

    if constrain_ends and len(path) > 1:
        x_end = path[-1][0]
        if abs(x_end) > 1e-9:
            scale = float(distance) / float(x_end)
            path = [(x * scale, y, z) for x, y, z in path]

    return path




def calibrate_correction_factor(pi,pf,p0,chlen = 1, distance = 30):
    # calibrates the script to detect the wire at correct length. ouput in pixels per meter
    thestickyiesteststick = vector(pi, pf, distance)
    #thestickyiesteststick = vector.convertspace()
    pole = vector((0, 0, -1), (distance, 0, 0))
    polep = vector((0,0,0), (pole.mag(),p0[0]/10000,p0[1]/10000))
    dcmc2g = polep.convertspace(pole)
    polep.v2vxbidcm(dcmc2g)
    #now take the vector and conver to pixels/m
    m0 = thestickyiesteststick.mag()
    thestickyiesteststick.v2vxbidcm(dcmc2g)
    m1 = thestickyiesteststick.mag()
    cf = m0/m1
    return cf# in pixels per meter or pixels/1 meter so cf*1m = cf at [] pixels
def characteristic_length(points,i,chlen, distance=30):
    if not points:
        return float(chlen)

    z = abs(points[i][2]) if i < len(points) else abs(points[-1][2])
    base_depth = max(abs(distance), 1e-9)

    # Higher depth -> lower apparent size, so pixels-per-meter scales inversely with depth.
    depth_scale = base_depth / max(z, 1e-9)
    depth_scale = float(np.clip(depth_scale, 0.2, 5.0))
    return float(chlen * depth_scale)






'''
# Example usage / Demo
if __name__ == "__main__":
    # Create a blank image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)  # White background
    
    # Define some example points
    points = [
        (100, 300),
        (250, 150),
        (400, 200),
        (550, 100),
        (700, 250)
    ]
    
    # Draw vectors between points
    img, vectors_info = draw_vectors_between_points(img, points, 
                                                     color=(0, 200, 0), 
                                                     thickness=2,
                                                     show_info=True,
                                                     arrow=True)
    
    # Print vector information
    print("\n=== Vector Information ===")
    for i, info in enumerate(vectors_info):
        print(f"\nVector {i+1}:")
        print(f"  From: {info['from']} → To: {info['to']}")
        print(f"  Vector: {info['vector']}")
        print(f"  Length: {info['length']:.2f} pixels")
        print(f"  Angle: {info['angle']:.2f}°")
    
    # Calculate angles between consecutive vectors
    print("\n=== Angles Between Vectors ===")
    for i in range(len(vectors_info) - 1):
        v1 = vectors_info[i]['vector']
        v2 = vectors_info[i + 1]['vector']
        angle_between = calculate_angle_between_vectors(v1, v2)
        print(f"Angle between Vector {i+1} and Vector {i+2}: {angle_between:.2f}°")
    
    # Save the image
    cv2.imwrite('vector_visualization.png', img)
    print("\n✓ Image saved as 'vector_visualization.png'")
    
    # Display the image (will open in window, press any key to close)
    cv2.imshow('Vectors Between Points', img)
    print("Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 3D vector demo
    points_3d = [
        (0, 0, 0),
        (2, 1, 1),
        (4, 2, 0),
        (5, 3, 2),
    ]

    print("\n=== 3D Vector Information ===")
    try:
        vectors_3d = display_vectors_3d(points_3d, show_labels=True, show_points=True, 
                                        title="3D Vector Demo", show_2d_projection=True)
        for i, info in enumerate(vectors_3d):
            print(f"Vector {i+1}: {info['vector']}, Length: {info['length']:.2f}")

        for i in range(len(vectors_3d) - 1):
            a3d = angle_between_vectors_3d(vectors_3d[i]["vector"], vectors_3d[i + 1]["vector"])
            print(f"Angle between 3D Vector {i+1} and {i+2}: {a3d:.2f}°")
    except ImportError as exc:
        print(f"3D display skipped: {exc}")'''

