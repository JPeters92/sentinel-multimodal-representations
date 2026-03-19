#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =====================
# CONFIG
# =====================
selected_cubes = ["043","027","442","053","034","022","368","348","356","449","212","228"]

s2_template   = "/net/data/deepfeatures/trainingcubes/000{cube}.zarr"
feat_template = "/net/data_ssd/deepfeatures/trainingcubes_processed/s1_s2_000{cube}.zarr"

# Output
out_dir = "paper_vis"
os.makedirs(out_dir, exist_ok=True)
pdf_path = os.path.join(out_dir, "S1_S2_PCA.pdf")

DPI = 300  # high DPI for paper

# Display stretch (for visualization only)
S2_PMIN,  S2_PMAX,  S2_GAMMA  = 2, 98, 1.6   # brighten S2 a bit
PCA_PMIN, PCA_PMAX, PCA_GAMMA = 1, 99, 1.2   # "soft" PCA colors

# Layout: 4 pairs (S2|PCA) per row, 3 rows => 12 cubes total
PAIRS_PER_ROW = 4
ROWS = int(np.ceil(len(selected_cubes) / PAIRS_PER_ROW))

# =====================
# HELPERS
# =====================
def pick_best_feature_time(feats: xr.DataArray) -> int:
    valid = feats.notnull().any("feature")
    counts = valid.sum(("y", "x"))
    return int(counts.argmax().values)

def stretch01(img, pmin=2, pmax=98, gamma=1.0):
    """Percentile stretch to [0,1], then gamma (display only)."""
    lo, hi = np.nanpercentile(img, [pmin, pmax])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.clip(img, 0, 1)
        return out ** (1.0 / gamma) if gamma != 1.0 else out
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return out ** (1.0 / gamma) if gamma != 1.0 else out

def make_s2_rgb(s2_da: xr.DataArray, t_idx: int) -> np.ndarray:
    """S2 RGB (B04,B03,B02) at time index t_idx (band,y,x), cropped by 7 px border."""
    bands = ["B04", "B03", "B02"]
    if not all(b in s2_da.band.values for b in bands):
        raise ValueError("Missing B04/B03/B02 in s2l2a bands.")
    frame = s2_da.sel(band=bands).isel(time=t_idx).load()  # (band,y,x)
    rgb  = frame.transpose("y", "x", "band").values.astype(np.float32)
    # remove 7-pixel border to match features
    border = 7
    rgb = rgb[border:-border, border:-border, :]
    # robust per-channel stretch for display
    for c in range(3):
        rgb[..., c] = stretch01(rgb[..., c], S2_PMIN, S2_PMAX, S2_GAMMA)
    return rgb

def make_pca_rgb(feats: xr.DataArray, t_idx: int) -> np.ndarray:
    """PCA (across features) at time t_idx -> (Y,X,3) RGB soft look."""
    frame = feats.isel(time=t_idx).load()   # (feature,y,x)
    F, Y, X = frame.shape
    data = frame.values.reshape(F, Y*X).T   # (pixels, features)
    mask = np.any(np.isnan(data), axis=1)
    valid = data[~mask]
    if valid.size == 0:
        raise ValueError("No valid pixels for PCA.")
    scaled = StandardScaler().fit_transform(valid)
    pc = PCA(n_components=3).fit_transform(scaled)  # (Nvalid,3)

    img = np.full((Y*X, 3), np.nan, dtype=np.float32)
    img[~mask] = pc[:, :3]
    img = img.reshape(Y, X, 3)

    # soft display: per-channel z-score -> percentile stretch -> gentle gamma
    for c in range(3):
        ch = img[..., c]
        mu = np.nanmean(ch); sd = np.nanstd(ch); sd = 1.0 if sd == 0 or not np.isfinite(sd) else sd
        z = (ch - mu) / sd
        img[..., c] = stretch01(z, PCA_PMIN, PCA_PMAX, PCA_GAMMA)
    return img

# =====================
# RENDER TO A SINGLE PDF
# =====================
with PdfPages(pdf_path) as pdf:
    # A4 landscape to fit 4 pairs per row
    fig = plt.figure(figsize=(11.0, 4.8), dpi=DPI)

    # Grid: (S2, PCA, spacer) x3 + (S2, PCA)
    # -> larger gaps between pairs, smaller between rows
    width_ratios = [1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1]

    gs = GridSpec(
        ROWS,
        len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.01,
        hspace=0.001,
    left=0.0, right=1.0, bottom=0.0, top=1.0   # <- NEU: keine äußeren Ränder
    )



    idx = 0
    for r in range(ROWS):
        pair_starts = [0, 3, 6, 9]
        for p in range(PAIRS_PER_ROW):
            if idx >= len(selected_cubes):
                break
            cube = selected_cubes[idx]
            s2_path   = s2_template.format(cube=cube)
            feat_path = feat_template.format(cube=cube)

            # Defaults for labels
            date_str = "n/a"
            lat = lon = np.nan
            s2_rgb = pca_rgb = None

            try:
                s2_ds   = xr.open_zarr(s2_path,   consolidated=True)
                feat_ds = xr.open_zarr(feat_path, consolidated=True)
                if "s2l2a" in s2_ds and "cloud_mask" in s2_ds and "features" in feat_ds:
                    # center coords
                    if "center_wgs84" in s2_ds.attrs:
                        lat, lon = s2_ds.attrs["center_wgs84"]
                    elif hasattr(s2_ds, "center_wgs84"):
                        lat, lon = s2_ds.center_wgs84

                    feats = feat_ds["features"]
                    t_idx = pick_best_feature_time(feats)
                    t_val = feats.time.values[t_idx]
                    date_str = str(np.datetime_as_string(t_val, unit="D"))

                    s2 = s2_ds.s2l2a.where(s2_ds.cloud_mask == 0)
                    s2_times = s2.time.values
                    match = np.where(s2_times == t_val)[0]
                    s2_idx = int(match[0]) if match.size else int(np.argmin(np.abs(s2_times - t_val)))

                    s2_rgb  = make_s2_rgb(s2,   s2_idx)
                    pca_rgb = make_pca_rgb(feats, t_idx)
            except Exception as e:
                print(f"⚠️ Skipping cube {cube}: {e}")

            # Place into grid
            c0 = pair_starts[p]
            ax_s2  = fig.add_subplot(gs[r, c0])
            ax_pca = fig.add_subplot(gs[r, c0+1])

            for ax in (ax_s2, ax_pca):
                ax.axis("off")

            if s2_rgb is not None:
                ax_s2.imshow(s2_rgb)
            if pca_rgb is not None:
                ax_pca.imshow(pca_rgb)

            # ---- Single-line, centered caption ABOVE the pair (date + location only) ----
            # Find the center x between the two axes; y slightly above them.
            bbox_s2 = ax_s2.get_position()
            bbox_pca = ax_pca.get_position()
            x_center = 0.5 * (bbox_s2.x0 + bbox_pca.x1)
            y_above = min(bbox_s2.y1, bbox_pca.y1) + 0.008  # klein +, aber < 1.0 bleiben
            caption = f"{date_str} — lat={lat:.4f}, lon={lon:.4f}"
            fig.text(x_center, y_above, caption, ha="center", va="bottom", fontsize=7)

            idx += 1

    pdf.savefig(fig, pad_inches=0)
    plt.close(fig)

print(f"✅ PDF saved: {pdf_path}")
