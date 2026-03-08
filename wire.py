import cv2
import numpy as np
import math

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
    Draw 2D X-Y projection of 3D vectors.
    
    Args:
        ax: Matplotlib axis to draw on.
        points_3d: List of (x, y, z) tuples.
        vectors_info: List of vector information dicts.
        show_labels: Whether to show length labels.
        show_points: Whether to show point markers.
    """
    xs = [p[0] for p in points_3d]
    ys = [p[1] for p in points_3d]

    if show_points:
        ax.scatter(xs, ys, c="tab:blue", s=35, label="Points", zorder=3)
        for i, p in enumerate(points_3d):
            ax.text(p[0], p[1], f"P{i}", fontsize=8, ha='right')

    # Draw vectors
    vector_ends_x = []
    vector_ends_y = []
    
    for i, info in enumerate(vectors_info):
        p1 = info['from']
        p2 = info['to']
        
        # Draw arrow
        ax.annotate('', xy=(p2[0], p2[1]), xytext=(p1[0], p1[1]),
                   arrowprops=dict(arrowstyle='->', color='tab:green', lw=2))
        
        vector_ends_x.append(p2[0])
        vector_ends_y.append(p2[1])
        
        if show_labels:
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            ax.text(mid_x, mid_y, f"L={info['length']:.2f}", 
                   color="tab:red", fontsize=8, ha='center')
    
    # Draw red dots at vector endpoints
    if vector_ends_x:
        ax.scatter(vector_ends_x, vector_ends_y, c="red", s=50, 
                  label="Vector ends", marker='o', zorder=4)
    
    ax.set_title("2D Projection (X-Y)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
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
        else:
            raise ValueError("Can only add another vector")
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
                vx = newspace.vector  / np.linalg.norm(np.array(newspace.vector))
                v =self.vector / np.linalg.norm(np.array(self.vector))
                
                axis  = np.cross(v, vx)
                angle = np.arccos(np.clip(np.dot(v, vx), -1, 1))
                
                # Rodrigues' rotation formula
                K = np.array([[0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0]])
                
                dcm = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
                self.dcm = dcm
                return self.dcm        
    def v2vxbidcm(self):
        directioncosmat = self.dcm
        v = list(self.vector)
        self.vector = directioncosmat*v

    
        

def sort_points(points):
    #sort points by y value hight o low
    return sorted(points, key=lambda p: p[1], reverse=True)

def wire_shape(points, fit_equation, chlen = 1, cfcl = 1, distance=30):
    points = sort_points(points)
    new = []
    new.append((points[0][0], points[0][1], distance))
    for i in(range(len(points) - 1)):
        cfcl = characteristic_length(new, i,chlen, distance=30)
        v= vector(points[i], points[i+1])
        theta = np.arccos(np.sqrt((chlen*cfcl)**2 - v.mag()**2)/cfcl) if v.length < chlen else 0
        #adding z to picture
        z = new[-1][2] - chlen * np.cos(theta)
        new.append((v.p2[0], v.p2[1], z))
    display_vectors_3d(new, show_labels=False, show_points=False, title="Wire Shape 3D Visualization", show_2d_projection=True)
    



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
def characteristic_length(points,i,chlen, distance):
    if i > 0:
        z = points[i-1][2]/30
        print(z)
    else:
        z = distance/30
    if z == 0:
        return chlen
    cfcf = distance/z
    print(cfcf)
    return chlen*cfcf#pixels or pixels /1m ----> mult by mag of the vector in vector shape






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

