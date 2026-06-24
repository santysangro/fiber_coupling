from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic
import umap

DATA_ROOT = Path(r"C:\Users\eqela\Desktop\fiber_coupling\Data")

DAYS = [
    "2026-06-12",
    "2026-06-13",
    "2026-06-14",
    "2026-06-15",
    "2026-06-16",
]

ACTUATOR_COLS = ["m0", "m1", "m2", "m3", "z"]
VOLTAGE_COL = "voltage_mV"

BROAD_PATTERN = "**/broad_scan/datasets/broad_global_scan_dataset.csv"
REGION_PATTERN = "**/medium/region_*/datasets/global_scan_dataset.csv"

OUTPUT_DIR = Path("fc_landscape_analysis_1")
OUTPUT_DIR.mkdir(exist_ok=True)

ABS_THRESHOLD_MV = 1000.0
REL_THRESHOLD_FRACTION = 0.20


def load_all_data():
    dfs = []

    for day in DAYS:
        day_dir = DATA_ROOT / day
        if not day_dir.exists():
            print(f"Missing day folder: {day_dir}")
            continue

        broad_files = list(day_dir.glob(BROAD_PATTERN))
        region_files = list(day_dir.glob(REGION_PATTERN))

        print(f"{day} | broad:  found {len(broad_files)} files")
        print(f"{day} | region: found {len(region_files)} files")

        for csv_path in broad_files:
            df = pd.read_csv(csv_path)
            df["dataset_type"] = "broad"
            df["source_file"] = str(csv_path)
            df["day"] = day
            dfs.append(df)

        for csv_path in region_files:
            df = pd.read_csv(csv_path)
            df["dataset_type"] = "region"
            df["source_file"] = str(csv_path)
            df["day"] = day
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No CSV files found. Check DATA_ROOT and patterns.")

    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna(subset=ACTUATOR_COLS + [VOLTAGE_COL])
    data = data[np.isfinite(data[ACTUATOR_COLS + [VOLTAGE_COL]]).all(axis=1)]

    print("\nLoaded datasets:")
    print(data["dataset_type"].value_counts())
    print(f"Loaded total points: {len(data)}")
    print(f"Voltage range: {data[VOLTAGE_COL].min():.2f} to {data[VOLTAGE_COL].max():.2f} mV")

    return data


def get_high_coupling_points(data):
    vmax = data[VOLTAGE_COL].max()
    relative_threshold = REL_THRESHOLD_FRACTION * vmax
    threshold = max(ABS_THRESHOLD_MV, relative_threshold)

    high = data[data[VOLTAGE_COL] >= threshold].copy()

    print("\nHigh-coupling filter:")
    print(f"Absolute threshold: {ABS_THRESHOLD_MV:.2f} mV")
    print(f"Relative threshold: {REL_THRESHOLD_FRACTION:.2f} * {vmax:.2f} = {relative_threshold:.2f} mV")
    print(f"Used threshold: {threshold:.2f} mV")
    print(f"High-coupling points: {len(high)} / {len(data)}")
    print("High-coupling points by dataset type:")
    print(high["dataset_type"].value_counts())

    if len(high) < 5:
        raise RuntimeError("Too few high-coupling points. Lower ABS_THRESHOLD_MV or REL_THRESHOLD_FRACTION.")

    high.to_csv(OUTPUT_DIR / "high_coupling_points.csv", index=False)
    return high, threshold

def plot_binned_max_surface(high, xcol="m0", ycol="m2", bins=50):

    x = high[xcol].to_numpy()
    y = high[ycol].to_numpy()
    v = high[VOLTAGE_COL].to_numpy()

    stat, xedge, yedge, _ = binned_statistic_2d(
        x,
        y,
        v,
        statistic="max",
        bins=bins,
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(
        stat.T,
        origin="lower",
        aspect="auto",
        extent=[
            xedge[0],
            xedge[-1],
            yedge[0],
            yedge[-1],
        ],
    )

    plt.colorbar(label="Maximum voltage [mV]")

    plt.scatter(
        x,
        y,
        c="k",
        s=3,
        alpha=0.15,
    )

    plt.xlabel(f"{xcol} position [steps]")
    plt.ylabel(f"{ycol} position [steps]")
    plt.title(f"Maximum coupling map: V({xcol}, {ycol})")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"max_coupling_map_{xcol}_{ycol}.png",
        dpi=300,
    )
    plt.close()


def plot_pair_fit(high, xcol, ycol, filename):
    x = high[xcol].to_numpy().reshape(-1, 1)
    y = high[ycol].to_numpy()
    voltage = high[VOLTAGE_COL].to_numpy()

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)

    slope = model.coef_[0]
    intercept = model.intercept_

    x_line = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(high[xcol], high[ycol], c=voltage, s=35, alpha=0.85)
    plt.plot(x_line.ravel(), y_line, linewidth=2)
    plt.colorbar(sc, label="Voltage [mV]")
    plt.xlabel(f"{xcol} position [steps]")
    plt.ylabel(f"{ycol} position [steps]")
    plt.title(f"High-coupling relation: {ycol} vs {xcol}\n{ycol} = {slope:.3f} {xcol} + {intercept:.1f}, R² = {r2:.3f}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

    return {
        "x": xcol,
        "y": ycol,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "n_points": len(high),
    }


def make_pair_plots(high):
    results = []
    results.append(plot_pair_fit(high, "m0", "m1", "high_coupling_m0_m1_linear_fit.png"))
    results.append(plot_pair_fit(high, "m2", "m3", "high_coupling_m2_m3_linear_fit.png"))

    fit_df = pd.DataFrame(results)
    fit_df.to_csv(OUTPUT_DIR / "high_coupling_linear_fit_results.csv", index=False)

    print("\nLinear fits for high-coupling manifold:")
    print(fit_df.round(4))


def make_3d_pca(high):
    X = high[ACTUATOR_COLS].to_numpy()
    voltage = high[VOLTAGE_COL].to_numpy()

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    pca = PCA(n_components=5)
    Xp = pca.fit_transform(Xn)

    explained = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_.T,
        index=ACTUATOR_COLS,
        columns=[f"PC{i+1}" for i in range(5)],
    )

    print("\nPCA explained variance on high-coupling points:")
    for i, val in enumerate(explained, start=1):
        print(f"PC{i}: {100 * val:.2f}%")
    print(f"PC1+PC2+PC3: {100 * explained[:3].sum():.2f}%")

    print("\nPCA loadings:")
    print(loadings.round(3))

    pca_df = high.copy()
    for i in range(5):
        pca_df[f"PC{i+1}"] = Xp[:, i]
    pca_df.to_csv(OUTPUT_DIR / "high_coupling_points_with_pca.csv", index=False)
    loadings.to_csv(OUTPUT_DIR / "high_coupling_pca_loadings.csv")

    # 3D PCA colored by voltage
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], c=voltage, s=35, alpha=0.85)
    ax.set_xlabel(f"PC1 [{100*explained[0]:.1f}%]")
    ax.set_ylabel(f"PC2 [{100*explained[1]:.1f}%]")
    ax.set_zlabel(f"PC3 [{100*explained[2]:.1f}%]")
    ax.set_title("3D PCA of high-coupling optimum manifold")
    fig.colorbar(sc, ax=ax, shrink=0.65, label="Voltage [mV]")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "high_coupling_3d_pca_voltage.png", dpi=300)
    plt.close()

    # 3D PCA colored by dataset type
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    for dtype in sorted(pca_df["dataset_type"].unique()):
        sub = pca_df[pca_df["dataset_type"] == dtype]
        ax.scatter(sub["PC1"], sub["PC2"], sub["PC3"], s=35, alpha=0.85, label=dtype)
    ax.set_xlabel(f"PC1 [{100*explained[0]:.1f}%]")
    ax.set_ylabel(f"PC2 [{100*explained[1]:.1f}%]")
    ax.set_zlabel(f"PC3 [{100*explained[2]:.1f}%]")
    ax.set_title("3D PCA of high-coupling points by dataset type")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "high_coupling_3d_pca_dataset_type.png", dpi=300)
    plt.close()
    plot_pca_explained_variance(explained)

    return pca_df, scaler, pca


def plot_reduced_surface(high, xcol="m0", ycol="m2"):
    """
    Plot coupling voltage as an 2D/3D surface over reduced coordinates.
    """

    x = high[xcol].to_numpy()
    y = high[ycol].to_numpy()
    v = high[VOLTAGE_COL].to_numpy()

    xi = np.linspace(x.min(), x.max(), 180)
    yi = np.linspace(y.min(), y.max(), 180)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), v, (X, Y), method="linear")

    plt.figure(figsize=(8, 6))
    im = plt.contourf(X, Y, Z, levels=50)
    sc = plt.scatter(x, y, c=v, s=12, edgecolors="k", linewidths=0.2, alpha=0.75)
    plt.colorbar(im, label="Interpolated voltage [mV]")
    plt.xlabel(f"{xcol} position [steps]")
    plt.ylabel(f"{ycol} position [steps]")
    plt.title(f"Reduced coupling surface: V({xcol}, {ycol})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"reduced_surface_{xcol}_{ycol}_contour.png", dpi=300)
    plt.close()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, alpha=0.75, linewidth=0, antialiased=True)
    ax.scatter(x, y, v, s=8, alpha=0.55)

    ax.set_xlabel(f"{xcol} [steps]")
    ax.set_ylabel(f"{ycol} [steps]")
    ax.set_zlabel("Voltage [mV]")
    ax.set_title(f"3D reduced coupling surface: V({xcol}, {ycol})")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"reduced_surface_{xcol}_{ycol}_3d.png", dpi=300)
    plt.close()

    print(f"\nReduced surface plots saved for V({xcol}, {ycol}).")



def make_umap(high):
    if not HAS_UMAP:
        print("\nUMAP not installed. Skipping UMAP plots.")
        print("Install with: pip install umap-learn")
        return

    X = high[ACTUATOR_COLS].to_numpy()
    voltage = high[VOLTAGE_COL].to_numpy()

    Xn = StandardScaler().fit_transform(X)

    n_neighbors = min(15, max(2, len(high) - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.05,
        random_state=0,
    )

    embedding = reducer.fit_transform(Xn)

    umap_df = high.copy()
    umap_df["UMAP1"] = embedding[:, 0]
    umap_df["UMAP2"] = embedding[:, 1]
    umap_df.to_csv(OUTPUT_DIR / "high_coupling_points_with_umap.csv", index=False)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=voltage, s=35, alpha=0.85)
    plt.colorbar(sc, label="Voltage [mV]")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP projection of high-coupling manifold")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "high_coupling_umap_voltage.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for dtype in sorted(umap_df["dataset_type"].unique()):
        sub = umap_df[umap_df["dataset_type"] == dtype]
        plt.scatter(sub["UMAP1"], sub["UMAP2"], s=35, alpha=0.85, label=dtype)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP projection by dataset type")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "high_coupling_umap_dataset_type.png", dpi=300)
    plt.close()

    print("\nUMAP plots saved.")


def plot_pca_explained_variance(explained):
    explained_pct = explained * 100
    labels = [f"PC{i}" for i in range(1, len(explained_pct) + 1)]

    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(labels, explained_pct)

    plt.ylabel("Explained variance [%]")
    plt.xlabel("Principal component")
    plt.title("PCA explained variance")

    plt.ylim(0, max(explained_pct) * 1.18)
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, explained_pct):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_explained_variance_clean.png", dpi=300)
    plt.close()

    print("\nClean PCA explained variance plot saved.")


def main():
    data = load_all_data()
    high, threshold = get_high_coupling_points(data)

    #make_pair_plots(high)
    #make_3d_pca(high)
    #make_umap(high)
    #plot_binned_max_surface(high, "m0", "m2")
    #plot_binned_max_surface(high, "m0", "z")
    #plot_binned_max_surface(high, "m2", "z")
    #threshold = np.percentile(high["voltage_mV"], 95)

    #best = high[high["voltage_mV"] >= threshold]
    #plot_z_voltage_trend(best)
    print(f"\nDone. Results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
