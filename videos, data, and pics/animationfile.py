"""Math-first animation of line sag reconstruction pipeline.

Sequence:
1) Poles + wire
2) Camera rays from wire to camera
3) Camera-plane normal
4) DCM/world->camera basis visualization + plane intersections
5) Vector addition along a parabolic arc toward the second pole

Run:
	python animationfile.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


def unit(v):
	n = np.linalg.norm(v)
	return v / max(n, 1e-12)


def rodrigues_align(a, b):
	"""Rotation matrix that maps unit vector a to unit vector b."""
	a = unit(np.asarray(a, dtype=float))
	b = unit(np.asarray(b, dtype=float))
	v = np.cross(a, b)
	c = float(np.dot(a, b))
	s = float(np.linalg.norm(v))
	if s < 1e-12:
		if c > 0:
			return np.eye(3)
		# 180-degree flip: pick arbitrary orthogonal axis.
		axis = unit(np.cross(a, np.array([1.0, 0.0, 0.0])))
		if np.linalg.norm(axis) < 1e-12:
			axis = unit(np.cross(a, np.array([0.0, 1.0, 0.0])))
		x, y, z = axis
		K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
		return np.eye(3) + 2.0 * (K @ K)
	K = np.array(
		[[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
		dtype=float,
	)
	return np.eye(3) + K + (K @ K) * ((1.0 - c) / (s * s))


def main():
	out_dir = Path(__file__).resolve().parent / "benchmark_outputs"
	out_dir.mkdir(parents=True, exist_ok=True)

	# Geometry setup.
	pole0_top = np.array([0.0, 0.0, 0.0])
	pole1_top = np.array([10.0, 0.0, 0.0])
	pole0_base = np.array([0.0, 0.0, -4.0])
	pole1_base = np.array([10.0, 0.0, -4.0])
	camera = pole0_top + np.array([0.0, 0.0, 0.9])  # Camera mounted above pole top.

	x = np.linspace(0.0, 10.0, 31)
	sag = 1.8
	z = -4.0 * sag * (x / 10.0) * (1.0 - x / 10.0)
	y = np.zeros_like(x)
	wire = np.column_stack([x, y, z])

	# Camera plane model.
	look_dir = unit(np.array([1.0, 0.0, -0.35]))
	plane_dist = 1.6
	plane_center = camera + plane_dist * look_dir

	up_guess = np.array([0.0, 0.0, 1.0])
	u_axis = unit(np.cross(up_guess, look_dir))
	if np.linalg.norm(u_axis) < 1e-8:
		u_axis = np.array([0.0, 1.0, 0.0])
	v_axis = unit(np.cross(look_dir, u_axis))

	# DCM: world x-axis mapped to camera normal direction (for visualization).
	dcm = rodrigues_align(np.array([1.0, 0.0, 0.0]), look_dir)

	fig = plt.figure(figsize=(16, 9), facecolor="#0D1020")
	ax_info = fig.add_axes([0.02, 0.06, 0.34, 0.88])
	ax = fig.add_axes([0.38, 0.05, 0.60, 0.90], projection="3d")

	phase_len = 70
	total_frames = phase_len * 5

	def style_axes():
		ax.set_facecolor("#0D1020")
		ax.xaxis.set_pane_color((0.08, 0.10, 0.18, 1.0))
		ax.yaxis.set_pane_color((0.08, 0.10, 0.18, 1.0))
		ax.zaxis.set_pane_color((0.08, 0.10, 0.18, 1.0))
		ax.tick_params(colors="#BDD5FF")
		ax.xaxis.label.set_color("#E7F0FF")
		ax.yaxis.label.set_color("#E7F0FF")
		ax.zaxis.label.set_color("#E7F0FF")
		ax.grid(color="#243158", alpha=0.14, linewidth=0.6)

	def style_info_axes():
		ax_info.set_facecolor("#0D1020")
		ax_info.set_xlim(0, 1)
		ax_info.set_ylim(0, 1)
		ax_info.axis("off")

	def draw_left_panel(phase, local_t):
		# Header and stage text.
		stage_labels = {
			0: "Stage 1: Wire Geometry",
			1: "Stage 2: Camera Rays",
			2: "Stage 3: Plane Normal",
			3: "Stage 4: DCM Mapping",
			4: "Stage 5: Vector Sum Arc",
		}
		ax_info.text(0.03, 0.96, "Math Visualization", color="#E7F0FF", fontsize=28, fontweight="bold", va="top")
		ax_info.text(0.03, 0.90, stage_labels.get(phase, "Stage"), color="#FFD67D", fontsize=20, fontweight="bold", va="top")

		# Video panel (left upper block).
		video_box = Rectangle((0.03, 0.50), 0.92, 0.34, linewidth=2.0, edgecolor="#4DD0FF", facecolor="#131B34")
		ax_info.add_patch(video_box)
		ax_info.text(0.05, 0.82, "Camera Feed (Module 1)", color="#A7C8FF", fontsize=15, fontweight="bold")

		# Simple 2D feed sketch: poles and sagging wire in image space.
		u = np.linspace(0.10, 0.88, 35)
		v = 0.68 + 0.10 * (4.0 * (u - 0.10) * (0.88 - u) / ((0.88 - 0.10) ** 2))
		k = max(2, int((0.2 + 0.8 * local_t) * len(u)))
		ax_info.plot([0.10, 0.10], [0.55, 0.78], color="#4DD0FF", linewidth=5)
		ax_info.plot([0.88, 0.88], [0.55, 0.78], color="#4DD0FF", linewidth=5)
		ax_info.plot(u[:k], v[:k], color="#FFC857", linewidth=4)
		ax_info.scatter([0.10], [0.82], color="#FF6B6B", s=90)

		# Matrix and equations panel (left lower block).
		math_box = Rectangle((0.03, 0.05), 0.92, 0.40, linewidth=2.0, edgecolor="#8FA8FF", facecolor="#101733")
		ax_info.add_patch(math_box)

		Rtxt = np.array2string(dcm, precision=2, suppress_small=True)
		ax_info.text(0.05, 0.40, "DCM Matrix R", color="#FFE29A", fontsize=19, fontweight="bold")
		ax_info.text(0.05, 0.32, f"R =\n{Rtxt}", color="#D8E5FF", fontsize=16, family="monospace", va="top")

		if phase == 2:
			ax_info.text(
				0.05,
				0.10,
				f"n = [{look_dir[0]:.3f}, {look_dir[1]:.3f}, {look_dir[2]:.3f}]^T",
				color="#FF9CB0",
				fontsize=18,
				fontweight="bold",
			)
		elif phase == 4:
			ax_info.text(0.05, 0.10, "p_{k+1} = p_k + delta v_k", color="#FFE29A", fontsize=22, fontweight="bold")
		else:
			ax_info.text(0.05, 0.10, "v_cam = R * v_world", color="#FFE29A", fontsize=20, fontweight="bold")

	def draw_common(progress_wire=1.0):
		# Poles.
		ax.plot([pole0_base[0], pole0_top[0]], [pole0_base[1], pole0_top[1]], [pole0_base[2], pole0_top[2]], color="#4DD0FF", linewidth=4)
		ax.plot([pole1_base[0], pole1_top[0]], [pole1_base[1], pole1_top[1]], [pole1_base[2], pole1_top[2]], color="#4DD0FF", linewidth=4)

		# Wire.
		n = max(2, int(progress_wire * len(wire)))
		ww = wire[:n]
		ax.plot(ww[:, 0], ww[:, 1], ww[:, 2], color="#FFC857", linewidth=3)

		# Camera point.
		ax.scatter([camera[0]], [camera[1]], [camera[2]], color="#FF6B6B", s=80)
		ax.text(camera[0], camera[1] + 0.2, camera[2] + 0.2, "Camera", color="#FFB0B0", fontsize=11)

	def ray_plane_intersection(ray_pt, ray_dir):
		denom = np.dot(ray_dir, look_dir)
		if abs(denom) < 1e-10:
			return None
		t = np.dot(plane_center - ray_pt, look_dir) / denom
		if t <= 0:
			return None
		return ray_pt + t * ray_dir

	def update(frame):
		ax_info.cla()
		style_info_axes()
		ax.cla()
		style_axes()
		ax.set_xlim(-0.8, 10.8)
		ax.set_ylim(-2.2, 2.2)
		ax.set_zlim(-4.6, 2.3)
		ax.set_xticks([0, 5, 10])
		ax.set_yticks([-2, 0, 2])
		ax.set_zticks([-4, -2, 0, 2])
		ax.set_xlabel("X", fontsize=12)
		ax.set_ylabel("Y", fontsize=12)
		ax.set_zlabel("Z", fontsize=12)
		ax.view_init(elev=22, azim=-66)

		phase = frame // phase_len
		local_t = (frame % phase_len) / (phase_len - 1)
		draw_left_panel(phase, local_t)

		if phase == 0:
			draw_common(progress_wire=local_t)
			ax.set_title("Poles and hanging wire model", color="#E7F0FF", fontsize=28)

		elif phase == 1:
			draw_common(progress_wire=1.0)
			k = max(2, int(local_t * len(wire)))
			sample = wire[::3][:k]
			for p in sample:
				ax.plot([p[0], camera[0]], [p[1], camera[1]], [p[2], camera[2]], color="#79D7FF", alpha=0.8, linewidth=1.4)
			ax.set_title("Segment projection rays back to camera", color="#E7F0FF", fontsize=28)

		elif phase == 2:
			draw_common(progress_wire=1.0)
			# Camera plane patch.
			pu = np.linspace(-1.0, 1.0, 8)
			pv = np.linspace(-0.7, 0.7, 8)
			UU, VV = np.meshgrid(pu, pv)
			PX = plane_center[0] + UU * u_axis[0] + VV * v_axis[0]
			PY = plane_center[1] + UU * u_axis[1] + VV * v_axis[1]
			PZ = plane_center[2] + UU * u_axis[2] + VV * v_axis[2]
			ax.plot_surface(PX, PY, PZ, alpha=0.22, color="#6BE6FF", linewidth=0)

			# True normal vector from plane definition (animated length only).
			norm_scale = 0.2 + 1.8 * local_t
			ax.quiver(
				camera[0],
				camera[1],
				camera[2],
				norm_scale * look_dir[0],
				norm_scale * look_dir[1],
				norm_scale * look_dir[2],
				color="#FF5E78",
				linewidth=2.5,
				arrow_length_ratio=0.14,
			)
			end = camera + norm_scale * look_dir
			ax.text(end[0], end[1], end[2], "n", color="#FF9CB0", fontsize=14)
			ax.set_title("Camera plane and computed normal vector", color="#E7F0FF", fontsize=28)

		elif phase == 3:
			draw_common(progress_wire=1.0)

			# World basis at camera.
			scale = 1.0
			ex = np.array([1.0, 0.0, 0.0])
			ey = np.array([0.0, 1.0, 0.0])
			ez = np.array([0.0, 0.0, 1.0])
			ax.plot([camera[0], camera[0] + scale * ex[0]], [camera[1], camera[1] + scale * ex[1]], [camera[2], camera[2] + scale * ex[2]], color="#FFD166", linewidth=2)
			ax.plot([camera[0], camera[0] + scale * ey[0]], [camera[1], camera[1] + scale * ey[1]], [camera[2], camera[2] + scale * ey[2]], color="#8CE99A", linewidth=2)
			ax.plot([camera[0], camera[0] + scale * ez[0]], [camera[1], camera[1] + scale * ez[1]], [camera[2], camera[2] + scale * ez[2]], color="#66B3FF", linewidth=2)

			# DCM-rotated basis.
			alpha = 0.25 + 0.75 * local_t
			rex = dcm @ ex
			rey = dcm @ ey
			rez = dcm @ ez
			ax.plot([camera[0], camera[0] + alpha * scale * rex[0]], [camera[1], camera[1] + alpha * scale * rex[1]], [camera[2], camera[2] + alpha * scale * rex[2]], color="#FF8B3D", linewidth=3)
			ax.plot([camera[0], camera[0] + alpha * scale * rey[0]], [camera[1], camera[1] + alpha * scale * rey[1]], [camera[2], camera[2] + alpha * scale * rey[2]], color="#44E4A8", linewidth=3)
			ax.plot([camera[0], camera[0] + alpha * scale * rez[0]], [camera[1], camera[1] + alpha * scale * rez[1]], [camera[2], camera[2] + alpha * scale * rez[2]], color="#6AB8FF", linewidth=3)

			# Project selected wire points onto camera plane.
			for p in wire[::5]:
				r = p - camera
				hit = ray_plane_intersection(camera, r)
				if hit is not None:
					ax.plot([camera[0], hit[0]], [camera[1], hit[1]], [camera[2], hit[2]], color="#5AD1FF", alpha=0.45, linewidth=1.1)
					ax.scatter([hit[0]], [hit[1]], [hit[2]], color="#B2F0FF", s=12)

			ax.set_title("DCM world-to-camera mapping and plane intersections", color="#E7F0FF", fontsize=28)

		else:
			draw_common(progress_wire=1.0)
			# Vector-addition build-up across parabola.
			k = max(2, int(local_t * len(wire)))
			pts = wire[:k]
			ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#FFD166", linewidth=3.2)
			ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="#E7F0FF", s=12)

			for i in range(len(pts) - 1):
				p0 = pts[i]
				p1 = pts[i + 1]
				ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="#FF6B6B", linewidth=1.8, alpha=0.9)

			ax.set_title("Vector addition reconstructs parabolic arc to Pole 2", color="#E7F0FF", fontsize=28)

	anim = FuncAnimation(fig, update, frames=total_frames, interval=40, blit=False)

	mp4_path = out_dir / "math_setup_animation.mp4"
	gif_path = out_dir / "math_setup_animation.gif"

	saved = None
	try:
		anim.save(mp4_path, dpi=120, fps=25)
		saved = mp4_path
	except Exception:
		anim.save(gif_path, dpi=100, fps=18)
		saved = gif_path

	plt.close(fig)
	print(f"Saved animation: {saved.resolve()}")


if __name__ == "__main__":
	main()
