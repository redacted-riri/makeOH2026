from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from testshape import (
    build_reference_parabola,
    estimate_pixels_per_meter,
    project_to_camera_view,
    verify_reconstruction,
)
from wire import wire_shape


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


def main():
    output_dir = Path("benchmark_outputs")
    output_dir.mkdir(exist_ok=True)

    spans = [100.0]
    # Requested sweep: sag 0..4 with dense intermediates, fixed camera height,
    # fixed tilt, and high-resolution pixel density range around 30 (+/-5).
    sags = np.linspace(0.0, 4.0, 81)
    tilts = np.linspace(20.0, 50.0, 31)
    heights = np.linspace(0.0, 3.0, 31)
    ppms = [30.0]
    num_points = 101

    results = []
    for span in spans:
        for sag in sags:
            for tilt in tilts:
                for h in heights:
                    for ppm in ppms:
                            results.append(run_case(span, float(sag), float(tilt), float(h), float(ppm), num_points=num_points))

    output_csv = output_dir / "parabola_benchmark_results.csv"
    save_results_csv(results, output_csv)
    save_error_plots(results, output_dir)
    save_avg_error_vs_camera_height(results, output_dir)

    mae_abs_vals = np.array([r["mae_abs"] for r in results], dtype=float)
    pct_errs = np.array([r["a_pct_error"] for r in results], dtype=float)
    pass_rate = float(np.mean(pct_errs <= 10.0) * 100.0)

    print(f"Ran {len(results)} cases.")
    print(f"Sweep config: sag=[0,4], sag_samples={len(sags)}, tilt=[{tilts[0]:.1f},{tilts[-1]:.1f}], tilt_samples={len(tilts)}, camera_height=[{heights[0]:.1f},{heights[-1]:.1f}], height_samples={len(heights)}, ppm={ppms[0]}, num_points={num_points}")
    print(f"Mean absolute error (m): {mae_abs_vals.mean():.4f}")
    print(f"Median a%% error: {np.median(pct_errs):.2f}%")
    print(f"<=10% a error pass rate: {pass_rate:.2f}%")

    # Tilt summary requested: average error per degree and best tilt with stdev.
    tilt_vals = sorted({float(r["tilt_deg"]) for r in results})
    tilt_summary = []
    for t in tilt_vals:
        vals = np.array([r["mae_abs"] for r in results if abs(float(r["tilt_deg"]) - t) < 1e-12], dtype=float)
        tilt_summary.append((t, float(vals.mean()), float(vals.std(ddof=0))))

    best_tilt, best_mean, best_std = min(tilt_summary, key=lambda x: x[1])
    print(f"Best tilt by mean absolute error: {best_tilt:.1f} deg (mean={best_mean:.4f} m, stdev={best_std:.4f} m)")

    print("Tilt mean absolute error summary (deg, mean_m, stdev_m):")
    for t, m, s in tilt_summary:
        print(f"  {t:.1f}, {m:.4f}, {s:.4f}")
    print(f"Saved: {output_csv}")
    print(f"Saved: {output_dir / 'error_vs_parameters.png'}")
    print(f"Saved: {output_dir / 'a_error_histogram.png'}")
    print(f"Saved: {output_dir / 'avg_error_vs_camera_height.png'}")


if __name__ == "__main__":
    main()