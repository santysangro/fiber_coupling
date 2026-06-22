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
# Optional UMAP. Script still works if umap-learn is not installed.
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# =========================
# USER SETTINGS
# =========================

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


# =========================
# DATA LOADING
# =========================

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


# =========================
# LINEAR PAIR PLOTS
# =========================

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


# =========================
# PCA 3D
# =========================

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


# =========================
# REDUCED MANIFOLD SURFACE PLOTS
# =========================

def plot_reduced_surface(high, xcol="m0", ycol="m2"):
    """
    Plot coupling voltage as an interpolated 2D/3D surface over reduced coordinates.

    Since m1 is strongly correlated with m0 and m3 is strongly correlated with m2,
    the high-coupling manifold can be approximately viewed using m0, m2, and z.
    This function plots V(m0, m2).
    """

    x = high[xcol].to_numpy()
    y = high[ycol].to_numpy()
    v = high[VOLTAGE_COL].to_numpy()

    xi = np.linspace(x.min(), x.max(), 180)
    yi = np.linspace(y.min(), y.max(), 180)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), v, (X, Y), method="linear")

    # -------------------------
    # 2D interpolated contour
    # -------------------------
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

    # -------------------------
    # 3D surface
    # -------------------------
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


# =========================
# UMAP
# =========================

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

# =========================
# Z-AXIS / FOCUS TREND
# =========================
# =========================
# PCA EXPLAINED VARIANCE
# =========================

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

def make_combined_manifold_figure_4panel(high):
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    X = high[ACTUATOR_COLS].to_numpy()
    Xn = StandardScaler().fit_transform(X)

    pca = PCA(n_components=5)
    pca.fit(Xn)
    explained_pct = pca.explained_variance_ratio_ * 100
    loadings = pca.components_.T
    voltage = high[VOLTAGE_COL].to_numpy()

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1, 1, 0.055],
        height_ratios=[1, 1],
        wspace=0.32,
        hspace=0.42,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    caxV = fig.add_subplot(gs[0, 2])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    caxL = fig.add_subplot(gs[1, 2])

    cmap_voltage = "viridis"
    fit_color = "crimson"

    # ---------- A ----------
    x = high["m0"].to_numpy().reshape(-1, 1)
    y = high["m1"].to_numpy()
    model = LinearRegression().fit(x, y)
    r2 = r2_score(y, model.predict(x))

    x_line = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)

    sc = axA.scatter(
        high["m0"], high["m1"],
        c=voltage,
        cmap=cmap_voltage,
        s=7,
        alpha=0.55,
        rasterized=True,
    )
    axA.plot(x_line.ravel(), model.predict(x_line), color=fit_color, lw=2.4)

    axA.set_xlabel(r"$m_0$ [steps]")
    axA.set_ylabel(r"$m_1$ [steps]")
    axA.set_title(
        rf"$m_1 = {model.coef_[0]:.2f}m_0 {model.intercept_:+.0f}$"
        + "\n"
        + rf"$R^2 = {r2:.3f}$"
    )
    axA.grid(alpha=0.2)

    # ---------- B ----------
    x = high["m2"].to_numpy().reshape(-1, 1)
    y = high["m3"].to_numpy()
    model = LinearRegression().fit(x, y)
    r2 = r2_score(y, model.predict(x))

    x_line = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)

    axB.scatter(
        high["m2"], high["m3"],
        c=voltage,
        cmap=cmap_voltage,
        s=7,
        alpha=0.55,
        rasterized=True,
    )
    axB.plot(x_line.ravel(), model.predict(x_line), color=fit_color, lw=2.4)

    axB.set_xlabel(r"$m_2$ [steps]")
    axB.set_ylabel(r"$m_3$ [steps]")
    axB.set_title(
        rf"$m_3 = {model.coef_[0]:.2f}m_2 {model.intercept_:+.0f}$"
        + "\n"
        + rf"$R^2 = {r2:.3f}$"
    )
    axB.grid(alpha=0.2)

    cbar = fig.colorbar(sc, cax=caxV)
    cbar.set_label("Voltage [mV]")

    # ---------- C ----------
    labels = [f"PC{i}" for i in range(1, 6)]
    bars = axC.bar(labels, explained_pct, edgecolor="black", linewidth=0.6)

    axC.set_ylabel("Explained variance [%]")
    axC.set_xlabel("Principal component")
    axC.set_title(rf"PC1--PC3 explain {explained_pct[:3].sum():.1f}%")
    axC.set_ylim(0, max(explained_pct) * 1.2)
    axC.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, explained_pct):
        axC.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # ---------- D ----------
    im = axD.imshow(
        loadings[:, :3],
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    axD.set_xticks([0, 1, 2])
    axD.set_xticklabels(["PC1", "PC2", "PC3"])
    axD.set_yticks(range(len(ACTUATOR_COLS)))
    axD.set_yticklabels([r"$m_0$", r"$m_1$", r"$m_2$", r"$m_3$", r"$z$"])
    axD.set_title("PCA loadings")

    for i in range(len(ACTUATOR_COLS)):
        for j in range(3):
            value = loadings[i, j]
            axD.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 else "black",
            )

    cbar = fig.colorbar(im, cax=caxL)
    cbar.set_label("Loading")

    # ---------- panel labels ----------
    for ax, label in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        ax.text(
            -0.13,
            1.10,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.suptitle(
        "Lower-dimensional structure of high-coupling solutions",
        fontsize=14,
        y=0.985,
    )

    fig.savefig(
        OUTPUT_DIR / "combined_high_coupling_manifold_4panel_professional.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "combined_high_coupling_manifold_4panel_professional.pdf",
        bbox_inches="tight",
    )

    plt.close()
    print("\nProfessional 4-panel manifold figure saved.")

# =========================
# MAIN
# =========================

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
    make_combined_manifold_figure_4panel(high)
    print(f"\nDone. Results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
