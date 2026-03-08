import math
import numpy as np
import matplotlib.pyplot as plt

from wire import wire_shape


def build_reference_parabola(span=100.0, sag=20.0, num_points=21):
	"""Create a parabola in X-Z with endpoints at z=0 and center at z=-sag."""
	xs = np.linspace(0.0, span, num_points)
	a = sag / ((span / 2.0) ** 2)
	zs = a * (xs - span / 2.0) ** 2 - sag
	return [(float(x), 0.0, float(z)) for x, z in zip(xs, zs)]


def project_to_camera_view(points_3d, pixels_per_meter=8.0, tilt_deg=35.0, origin=(120, 260), camera_height_m=0.0):
	"""
	Project 3D wire points to a 2D camera-style view.

	Camera geometry: top-down view with forward tilt so both X progression and Z sag
	affect image rows. This yields realistic non-uniform image spacing.
	"""
	tilt = math.radians(tilt_deg)
	# Higher camera height reduces apparent motion in image space.
	height_scale = 1.0 / (1.0 + max(camera_height_m, 0.0))
	camera_points = []
	for x, _, z in points_3d:
		# Horizontal image coordinate mostly follows wire span (x direction).
		u = origin[0] + pixels_per_meter * height_scale * x

		# Vertical image coordinate combines depth along span and sag (z).
		v = origin[1] + pixels_per_meter * height_scale * (x * math.sin(tilt) - z * math.cos(tilt))
		camera_points.append((float(u), float(v)))

	return camera_points


def estimate_pixels_per_meter(camera_points, reference_points):
	"""Estimate average pixels-per-meter from corresponding point spacings."""
	ratios = []
	for i in range(len(camera_points) - 1):
		du = camera_points[i + 1][0] - camera_points[i][0]
		dv = camera_points[i + 1][1] - camera_points[i][1]
		pixel_dist = math.sqrt(du * du + dv * dv)

		dx = reference_points[i + 1][0] - reference_points[i][0]
		dz = reference_points[i + 1][2] - reference_points[i][2]
		world_dist = math.sqrt(dx * dx + dz * dz)

		if world_dist > 1e-9:
			ratios.append(pixel_dist / world_dist)

	return float(np.median(ratios)) if ratios else 1.0


def verify_reconstruction(model_points, reconstructed_points):
	"""
	Compute absolute and shape-normalized RMSE in Z on the model's X grid.

	Shape-normalized RMSE allows global scale/offset differences caused by projection.
	"""
	model_x = np.array([p[0] for p in model_points], dtype=float)
	model_z = np.array([p[2] for p in model_points], dtype=float)

	rec_x = np.array([p[0] for p in reconstructed_points], dtype=float)
	rec_z = np.array([p[2] for p in reconstructed_points], dtype=float)

	# Compare on the model's x-grid.
	rec_z_interp = np.interp(model_x, rec_x, rec_z)
	rmse_abs = float(np.sqrt(np.mean((rec_z_interp - model_z) ** 2)))

	# Affine-align reconstructed z to model z to compare shape independent of scale/offset.
	design = np.vstack([rec_z_interp, np.ones_like(rec_z_interp)]).T
	alpha, beta = np.linalg.lstsq(design, model_z, rcond=None)[0]
	rec_z_aligned = alpha * rec_z_interp + beta
	rmse_shape = float(np.sqrt(np.mean((rec_z_aligned - model_z) ** 2)))

	a, b, c = np.polyfit(rec_x, rec_z, 2)
	return rmse_abs, rmse_shape, (float(a), float(b), float(c))


def plot_parabola_comparison(model_points, reconstructed_points, model_coeffs, reconstructed_coeffs):
	"""Plot model vs reconstructed parabola in X-Z and compare coefficients."""
	model_x = np.array([p[0] for p in model_points], dtype=float)
	model_z = np.array([p[2] for p in model_points], dtype=float)
	rec_x = np.array([p[0] for p in reconstructed_points], dtype=float)
	rec_z = np.array([p[2] for p in reconstructed_points], dtype=float)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

	# Left: X-Z plane comparison.
	ax1.plot(model_x, model_z, "o-", color="tab:blue", label="Model parabola")
	ax1.plot(rec_x, rec_z, "s-", color="tab:orange", label="Reconstructed parabola")
	ax1.set_title("Parabola Comparison in X-Z Plane")
	ax1.set_xlabel("X")
	ax1.set_ylabel("Z")
	ax1.grid(True, alpha=0.3)
	ax1.legend(loc="best")

	# Right: coefficient comparison.
	labels = ["a", "b", "c"]
	x = np.arange(len(labels))
	width = 0.36
	ax2.bar(x - width / 2, model_coeffs, width, label="Model", color="tab:blue")
	ax2.bar(x + width / 2, reconstructed_coeffs, width, label="Reconstructed", color="tab:orange")
	ax2.set_xticks(x)
	ax2.set_xticklabels(labels)
	ax2.set_title("Parabola Coefficient Comparison")
	ax2.set_ylabel("Coefficient Value")
	ax2.grid(True, axis="y", alpha=0.3)
	ax2.legend(loc="best")

	plt.tight_layout()
	plt.show()


def main():
	span = 100.0
	sag = 12.0
	num_points = 21
	camera_height_m = 1.0

	model = build_reference_parabola(span=span, sag=sag, num_points=num_points)
	camera_points = project_to_camera_view(
		model,
		pixels_per_meter=8.0,
		tilt_deg=35.0,
		camera_height_m=camera_height_m,
	)

	# Camera-view points are what the vision pipeline would observe.
	print("Camera-view points (u, v):")
	print([(round(u, 2), round(v, 2)) for u, v in camera_points])

	ppm = estimate_pixels_per_meter(camera_points, model)
	print(f"Estimated pixels_per_meter: {ppm:.4f}")
	print(f"Model config: span={span}, sag={sag}, points={num_points}, camera_height_m={camera_height_m}")

	reconstructed = wire_shape(
		camera_points,
		"parabola_up",
		chlen=ppm,
		distance=span,
		anchor_origin=True,
		constrain_ends=True,
		camera_params={
			"pixels_per_meter": 8.0,
			"tilt_deg": 35.0,
			"origin": (120, 260),
			"camera_height_m": camera_height_m,
		},
	)

	# Anchor model to same origin convention used by wire_shape(anchor_origin=True).
	x0, _, z0 = model[0]
	model_anchored = [(x - x0, 0.0, z - z0) for x, _, z in model]
	model_x = np.array([p[0] for p in model_anchored], dtype=float)
	model_z = np.array([p[2] for p in model_anchored], dtype=float)
	model_coeffs = tuple(float(v) for v in np.polyfit(model_x, model_z, 2))

	rmse_abs, rmse_shape, coeffs = verify_reconstruction(model_anchored, reconstructed)
	print(f"Reconstruction RMSE in z (absolute units): {rmse_abs:.4f}")
	print(f"Reconstruction RMSE in z (shape-normalized): {rmse_shape:.4f}")
	print(f"Reconstructed parabola coefficients (a,b,c): {coeffs}")
	print(f"Parabola opens upward: {coeffs[0] > 0}")
	print(f"Model parabola coefficients (a,b,c): {model_coeffs}")

	# Basic success criteria for this synthetic test.
	ok_shape = coeffs[0] > 0
	ok_fit = rmse_shape < 1.0
	print(f"Shape detection PASS: {ok_shape}")
	print(f"Model verification PASS (shape RMSE<1): {ok_fit}")

	plot_parabola_comparison(model_anchored, reconstructed, model_coeffs, coeffs)


if __name__ == "__main__":
	main()
