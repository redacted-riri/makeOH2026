from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from testshape import (
    build_reference_parabola,
    estimate_pixels_per_meter,
    project_to_camera_view,
    verify_reconstruction,
)
from wire import wire_shape


def save_sag_height_heatmap_from_csv(csv_path, output_png):
    """Create a sag vs camera-height heatmap from benchmark CSV using average error in meters."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            metric_key = "mae_abs" if "mae_abs" in r else ("avg_mae_m" if "avg_mae_m" in r else None)
            if "sag" in r and "camera_height_m" in r and metric_key is not None:
                rows.append((float(r["sag"]), float(r["camera_height_m"]), float(r[metric_key])))

    if not rows:
        raise ValueError(f"No usable rows found in {csv_path}")

    sags = sorted({r[0] for r in rows})
    heights = sorted({r[1] for r in rows})

    # Aggregate mean MAE for each (sag, height) cell.
    agg = {(s, h): [] for s in sags for h in heights}
    for s, h, e in rows:
        agg[(s, h)].append(e)

    grid = np.full((len(heights), len(sags)), np.nan, dtype=float)
    for iy, h in enumerate(heights):
        for ix, s in enumerate(sags):
            vals = agg[(s, h)]
            if vals:
                grid[iy, ix] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[min(sags), max(sags), min(heights), max(heights)],
        cmap="RdYlGn_r",
    )
    ax.set_title("Average Error Heatmap (m): Sag vs Camera Height")
    ax.set_xlabel("Sag")
    ax.set_ylabel("Camera Height (m)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Error (m)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def run_case(span, sag, tilt_deg, camera_height_m, pixels_per_meter, num_points=21):
    model = build_reference_parabola(span=span, sag=sag, num_points=num_points)
    camera_points = project_to_camera_view(
        model,
        pixels_per_meter=pixels_per_meter,
        tilt_deg=tilt_deg,
        origin=(120, 260),
        camera_height_m=camera_height_m,
    )

    ppm_est = estimate_pixels_per_meter(camera_points, model)
    reconstructed = wire_shape(
        camera_points,
        "parabola_up",
        chlen=ppm_est,
        distance=span,
        anchor_origin=True,
        constrain_ends=True,
        camera_params={
            "pixels_per_meter": pixels_per_meter,
            "tilt_deg": tilt_deg,
            "origin": (120, 260),
            "camera_height_m": camera_height_m,
        },
    )

    x0, _, z0 = model[0]
    model_anchored = [(x - x0, 0.0, z - z0) for x, _, z in model]
    rmse_abs, rmse_shape, coeffs = verify_reconstruction(model_anchored, reconstructed)

    model_x = np.array([p[0] for p in model_anchored], dtype=float)
    model_z = np.array([p[2] for p in model_anchored], dtype=float)
    rec_x = np.array([p[0] for p in reconstructed], dtype=float)
    rec_z = np.array([p[2] for p in reconstructed], dtype=float)
    rec_z_interp = np.interp(model_x, rec_x, rec_z)
    mae_abs = float(np.mean(np.abs(rec_z_interp - model_z)))

    model_a, model_b, model_c = np.polyfit(model_x, model_z, 2)

    # % error on quadratic steepness term (a).
    a_pct_error = abs(coeffs[0] - model_a) / max(abs(model_a), 1e-9) * 100.0

    return {
        "span": span,
        "sag": sag,
        "num_points": num_points,
        "tilt_deg": tilt_deg,
        "camera_height_m": camera_height_m,
        "pixels_per_meter": pixels_per_meter,
        "mae_abs": mae_abs,
        "rmse_abs": rmse_abs,
        "rmse_shape": rmse_shape,
        "model_a": float(model_a),
        "recon_a": float(coeffs[0]),
        "a_pct_error": float(a_pct_error),
    }


def save_results_csv(results, output_csv):
    header = [
        "span",
        "sag",
        "num_points",
        "tilt_deg",
        "camera_height_m",
        "pixels_per_meter",
        "mae_abs",
        "rmse_abs",
        "rmse_shape",
        "model_a",
        "recon_a",
        "a_pct_error",
    ]

    lines = [",".join(header)]
    for r in results:
        lines.append(
            f"{r['span']},{r['sag']},{r['num_points']},{r['tilt_deg']},{r['camera_height_m']},{r['pixels_per_meter']},"
            f"{r['mae_abs']},{r['rmse_abs']},{r['rmse_shape']},{r['model_a']},{r['recon_a']},{r['a_pct_error']}"
        )

    output_csv.write_text("\n".join(lines), encoding="utf-8")


def save_error_plots(results, output_dir):
    sags = np.array([r["sag"] for r in results], dtype=float)
    heights = np.array([r["camera_height_m"] for r in results], dtype=float)
    err_mae = np.array([r["mae_abs"] for r in results], dtype=float)
    err_pct = np.array([r["a_pct_error"] for r in results], dtype=float)

    # Use robust color limits so a few outliers do not wash out the full plot.
    cmin = float(np.percentile(err_pct, 5))
    cmax = float(np.percentile(err_pct, 95))
    if cmax <= cmin:
        cmax = cmin + 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    plots = [
        (sags, "Sag", axes[0]),
        (heights, "Camera Height (m)", axes[1]),
    ]

    for x, label, ax in plots:
        sc = ax.scatter(
            x,
            err_mae,
            c=err_pct,
            cmap="RdYlGn_r",
            s=24,
            alpha=0.75,
            vmin=cmin,
            vmax=cmax,
            edgecolors="none",
        )

        # Overlay mean trend per unique parameter value.
        ux = np.unique(x)
        mean_mae = np.array([err_mae[x == xv].mean() for xv in ux], dtype=float)
        ax.plot(ux, mean_mae, color="black", linewidth=1.6, label="Mean Abs Error")

        ax.set_xlabel(label)
        ax.set_ylabel("Average Error (m)")
        ax.set_title(f"Average Error vs {label}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    cbar = fig.colorbar(sc, ax=axes, shrink=0.95)
    cbar.set_label("Quadratic a Error (%)")
    fig.suptitle("Average Reconstruction Error (m) vs Parameters")
    fig.savefig(output_dir / "error_vs_parameters.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(err_pct, bins=20, color="tab:orange", edgecolor="black")
    ax2.set_title("Distribution of Quadratic a Error (%)")
    ax2.set_xlabel("a Error (%)")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "a_error_histogram.png", dpi=150)
    plt.close(fig2)


def save_avg_error_vs_camera_height(results, output_dir):
    heights = sorted({float(r["camera_height_m"]) for r in results})
    mean_err = []
    std_err = []

    for h in heights:
        vals = np.array([r["mae_abs"] for r in results if abs(float(r["camera_height_m"]) - h) < 1e-12], dtype=float)
        mean_err.append(float(vals.mean()))
        std_err.append(float(vals.std(ddof=0)))

    x = np.array(heights, dtype=float)
    y = np.array(mean_err, dtype=float)
    ystd = np.array(std_err, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color="tab:blue", linewidth=2, label="Mean absolute error")
    ax.fill_between(x, y - ystd, y + ystd, color="tab:blue", alpha=0.2, label="+/- 1 std dev")
    ax.set_title("Average Error vs Camera Height")
    ax.set_xlabel("Camera Height (m)")
    ax.set_ylabel("Average Error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "avg_error_vs_camera_height.png", dpi=150)
    plt.close(fig)


def apply_pixel_measurement_noise(camera_points, origin, max_pct, rng):
    """Apply clipped Gaussian pixel-scale error around the camera origin."""
    max_frac = max_pct / 100.0
    sigma = max_frac / 3.0  # ~99.7% inside +/- max_pct before clipping

    noisy = []
    for u, v in camera_points:
        eps_u = float(np.clip(rng.normal(0.0, sigma), -max_frac, max_frac))
        eps_v = float(np.clip(rng.normal(0.0, sigma), -max_frac, max_frac))
        du = u - origin[0]
        dv = v - origin[1]
        noisy.append((origin[0] + du * (1.0 + eps_u), origin[1] + dv * (1.0 + eps_v)))
    return noisy


def save_noise_contour_plots(agg_results, output_png):
    sags = sorted({r["sag"] for r in agg_results})
    noise_levels = sorted({r["noise_pct"] for r in agg_results})

    sag_pct = np.array([(s / 100.0) * 100.0 for s in sags], dtype=float)  # span=100 -> identical value
    noise_arr = np.array(noise_levels, dtype=float)

    mae_grid = np.zeros((len(noise_levels), len(sags)), dtype=float)
    pct_grid = np.zeros((len(noise_levels), len(sags)), dtype=float)

    lookup = {(r["noise_pct"], r["sag"]): r for r in agg_results}
    for i, n in enumerate(noise_levels):
        for j, s in enumerate(sags):
            row = lookup[(n, s)]
            mae_grid[i, j] = row["avg_mae_m"]
            pct_grid[i, j] = row["avg_a_pct_error"]

    X, Y = np.meshgrid(sag_pct, noise_arr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    if mae_grid.shape[0] >= 2 and mae_grid.shape[1] >= 2:
        c1 = axes[0].contourf(X, Y, mae_grid, levels=16, cmap="RdYlGn_r")
        axes[0].contour(X, Y, mae_grid, levels=8, colors="k", linewidths=0.5, alpha=0.6)
    else:
        c1 = axes[0].pcolormesh(X, Y, mae_grid, shading="auto", cmap="RdYlGn_r")
    axes[0].set_title("Avg Absolute Error (m)")
    axes[0].set_xlabel("Sag (% of span)")
    axes[0].set_ylabel("Pixel Error Max Width (%)")
    fig.colorbar(c1, ax=axes[0], label="Avg Error (m)")

    if pct_grid.shape[0] >= 2 and pct_grid.shape[1] >= 2:
        c2 = axes[1].contourf(X, Y, pct_grid, levels=16, cmap="RdYlGn_r")
        axes[1].contour(X, Y, pct_grid, levels=8, colors="k", linewidths=0.5, alpha=0.6)
    else:
        c2 = axes[1].pcolormesh(X, Y, pct_grid, shading="auto", cmap="RdYlGn_r")
    axes[1].set_title("Avg Quadratic a Error (%)")
    axes[1].set_xlabel("Sag (% of span)")
    axes[1].set_ylabel("Pixel Error Max Width (%)")
    fig.colorbar(c2, ax=axes[1], label="a Error (%)")

    fig.suptitle("Reconstruction Sensitivity to Pixel Measurement Error")
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def save_noise_aggregate_csv(rows, path):
    header = [
        "noise_pct",
        "camera_height_m",
        "sag",
        "sag_pct",
        "avg_mae_m",
        "avg_a_pct_error",
        "std_mae_m",
        "std_a_pct_error",
        "n",
    ]
    lines = [",".join(header)]
    for r in rows:
        lines.append(
            f"{r['noise_pct']},{r['camera_height_m']},{r['sag']},{r['sag_pct']},{r['avg_mae_m']},"
            f"{r['avg_a_pct_error']},{r['std_mae_m']},{r['std_a_pct_error']},{r['n']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_sag_vs_camera_height_heatmap(agg_rows, output_png):
    # Average over noise levels to isolate geometry sensitivity.
    grouped = {}
    for r in agg_rows:
        key = (float(r["camera_height_m"]), float(r["sag"]))
        grouped.setdefault(key, []).append(float(r["avg_mae_m"]))

    heights = sorted({k[0] for k in grouped})
    sags = sorted({k[1] for k in grouped})

    Z = np.zeros((len(heights), len(sags)), dtype=float)
    for i, h in enumerate(heights):
        for j, s in enumerate(sags):
            Z[i, j] = float(np.mean(grouped[(h, s)]))

    X, Y = np.meshgrid(np.array(sags, dtype=float), np.array(heights, dtype=float))

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    hm = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
    ax.set_title("Average Reconstruction Error Heatmap")
    ax.set_xlabel("Sag (m)")
    ax.set_ylabel("Camera Height (m)")
    cbar = fig.colorbar(hm, ax=ax)
    cbar.set_label("Average Error (m)")
    ax.grid(True, alpha=0.15)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main():
    output_dir = Path("benchmark_outputs")
    output_dir.mkdir(exist_ok=True)

    # Noise robustness experiment settings.
    span = 100.0
    sags = np.linspace(0.1, 4.0, 79)  # exclude sag=0 for contour analysis
    noise_levels_pct = [5.0]
    camera_heights_m = [0.8, 1.0, 1.2, 1.5]
    repeats = 300

    tilt_deg = 35.0
    pixels_per_meter = 30.0
    origin = (120.0, 260.0)
    num_points = 101

    rng = np.random.default_rng(20260308)

    raw_rows = []
    agg_rows = []

    for camera_height_m in camera_heights_m:
        for sag in sags:
            model = build_reference_parabola(span=span, sag=float(sag), num_points=num_points)
            camera_points = project_to_camera_view(
                model,
                pixels_per_meter=pixels_per_meter,
                tilt_deg=tilt_deg,
                origin=origin,
                camera_height_m=camera_height_m,
            )

            x0, _, z0 = model[0]
            model_anchored = [(x - x0, 0.0, z - z0) for x, _, z in model]
            model_x = np.array([p[0] for p in model_anchored], dtype=float)
            model_z = np.array([p[2] for p in model_anchored], dtype=float)
            model_a = float(np.polyfit(model_x, model_z, 2)[0])

            for noise_pct in noise_levels_pct:
                mae_vals = []
                aerr_vals = []

                for _ in range(repeats):
                    noisy_points = apply_pixel_measurement_noise(camera_points, origin, noise_pct, rng)
                    ppm_est = estimate_pixels_per_meter(noisy_points, model)

                    reconstructed = wire_shape(
                        noisy_points,
                        "parabola_up",
                        chlen=ppm_est,
                        distance=span,
                        anchor_origin=True,
                        constrain_ends=True,
                        camera_params={
                            "pixels_per_meter": pixels_per_meter,
                            "tilt_deg": tilt_deg,
                            "origin": origin,
                            "camera_height_m": camera_height_m,
                        },
                    )

                    rmse_abs, rmse_shape, coeffs = verify_reconstruction(model_anchored, reconstructed)

                    rec_x = np.array([p[0] for p in reconstructed], dtype=float)
                    rec_z = np.array([p[2] for p in reconstructed], dtype=float)
                    rec_z_interp = np.interp(model_x, rec_x, rec_z)
                    mae_abs = float(np.mean(np.abs(rec_z_interp - model_z)))
                    a_pct_error = abs(coeffs[0] - model_a) / max(abs(model_a), 1e-9) * 100.0

                    mae_vals.append(mae_abs)
                    aerr_vals.append(float(a_pct_error))

                    raw_rows.append({
                        "camera_height_m": float(camera_height_m),
                        "sag": float(sag),
                        "sag_pct": float((sag / span) * 100.0),
                        "noise_pct": float(noise_pct),
                        "mae_abs": mae_abs,
                        "a_pct_error": float(a_pct_error),
                    })

                agg_rows.append({
                    "noise_pct": float(noise_pct),
                    "camera_height_m": float(camera_height_m),
                    "sag": float(sag),
                    "sag_pct": float((sag / span) * 100.0),
                    "avg_mae_m": float(np.mean(mae_vals)),
                    "avg_a_pct_error": float(np.mean(aerr_vals)),
                    "std_mae_m": float(np.std(mae_vals, ddof=0)),
                    "std_a_pct_error": float(np.std(aerr_vals, ddof=0)),
                    "n": repeats,
                })

    # Save aggregate table and contour visualization.
    agg_csv = output_dir / "noise_benchmark_aggregate.csv"
    save_noise_aggregate_csv(agg_rows, agg_csv)
    contour_png = output_dir / "noise_error_contours.png"
    save_noise_contour_plots(agg_rows, contour_png)
    sag_height_png = output_dir / "sag_vs_camera_height_error_heatmap.png"
    save_sag_height_heatmap_from_csv(agg_csv, sag_height_png)

    all_mae = np.array([r["avg_mae_m"] for r in agg_rows], dtype=float)
    all_aerr = np.array([r["avg_a_pct_error"] for r in agg_rows], dtype=float)

    print(
        f"Noise benchmark complete: {len(sags)} sag levels x {len(camera_heights_m)} camera heights "
        f"x {len(noise_levels_pct)} noise levels x {repeats} runs"
    )
    print(f"Mean(avg MAE m) across grid: {all_mae.mean():.4f}")
    print(f"Mean(avg a% error) across grid: {all_aerr.mean():.2f}%")
    print("Per-noise average summary (noise%, avg_error_m, avg_a_pct_error):")
    for n in noise_levels_pct:
        rows = [r for r in agg_rows if abs(r["noise_pct"] - n) < 1e-12]
        print(f"  {n:.1f}, {np.mean([r['avg_mae_m'] for r in rows]):.4f}, {np.mean([r['avg_a_pct_error'] for r in rows]):.2f}")

    print(f"Saved: {agg_csv}")
    print(f"Saved: {contour_png}")
    print(f"Saved: {sag_height_png}")


if __name__ == "__main__":
    main()