#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========= USER CONFIG =========
s1_log_file = Path("/scratch/jpeters/DeepFeatures/lightning_logs/needed_s1_version_172/metrics.csv")
s2_log_file = Path("/scratch/jpeters/DeepFeatures/lightning_logs/needed_s2_version_131/metrics.csv")

s1_max_epoch = 65
s2_max_epoch = 154

# Center/surroundings mixing weights used to recover MAE (surroundings)
ALPHA_CENTER = 0.7446
ALPHA_SURR   = 1.0 - ALPHA_CENTER  # 0.2554



# ========= HELPERS =========
def load_and_prepare(log_file: Path, max_epoch: int) -> pd.DataFrame:
    df = pd.read_csv(log_file)

    # Keep only per-epoch summaries (lr is NaN), clip to desired max epoch
    df = df[df["lr"].isna()]
    df = df[df["epoch"] <= max_epoch].reset_index(drop=True)

    # Robustly choose totals (prefer *_total if present; else fall back to historical naming)
    val_total  = df["val_mae"]   #if "val_total"   in df.columns else df.get("val_mae", pd.Series(np.nan, index=df.index))
    train_total= df["train_mae"] #if "train_total" in df.columns else df.get("train_mae", pd.Series(np.nan, index=df.index))

    # Center MAE columns (validation uses 'val_loss' for center MAE in your logs)
    val_center   = df["val_loss"]
    train_center = df["train_center"]

    # --- Reconstruct surroundings MAE for validation & training ---
    # val_mae_sur = (val_total - a * val_center) / (1 - a)
    df["mae_surroundings"] = (val_total - ALPHA_CENTER * val_center) / ALPHA_SURR

    # train_mae_sur = (train_total - a * train_center) / (1 - a)
    df["train_mae_surroundings"] = (train_total - ALPHA_CENTER * train_center) / ALPHA_SURR

    # Combine rows per epoch (prefer row that contains the train_* metrics, if duplicated)
    def combine_rows(group):
        combined = group.iloc[0].copy()

        # Always take validation-side values from any row that has them (use first non-NaN)
        for k in ["val_loss", "val_sam", "val_ssim", "mae_surroundings"]:
            if k in group.columns:
                gk = group[k].dropna()
                if not gk.empty:
                    combined[k] = gk.iloc[0]

        # Prefer the row with training metrics present
        non_nan_row = group[group["train_center"].notna()]
        if not non_nan_row.empty:
            r = non_nan_row.iloc[0]
            for k in ["train_center", "train_mae", "train_sam", "train_ssim", "train_total", "train_mae_surroundings"]:
                if k in r:
                    combined[k] = r[k]

        return combined

    combined_df = df.groupby("epoch").apply(combine_rows).reset_index(drop=True)
    return combined_df

# ========= LOAD =========
s1 = load_and_prepare(s1_log_file, s1_max_epoch)
s2 = load_and_prepare(s2_log_file, s2_max_epoch)

# ========= STYLE =========
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 17,
    'legend.fontsize': 14,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17
})

# ========= PLOT =========
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 6), dpi=150, sharex=False)

# ---- Row 1: Central point ----
# (1,1) Sentinel-1 · MAE (Center)
ax = axes[0, 0]
ax.plot(s1["epoch"], s1["train_center"], linewidth=2, label="Train MAE (Center)")
ax.plot(s1["epoch"], s1["val_loss"],   linewidth=2, linestyle="--", label="Val MAE (Center)")
ax.set_title("Sentinel-1 · MAE (Center)")
ax.grid(True, alpha=0.3)
ax.legend()

# (1,2) Sentinel-2 · MAE (Center)
ax = axes[0, 1]
ax.plot(s2["epoch"], s2["train_center"], linewidth=2, label="Train MAE (Center)")
ax.plot(s2["epoch"], s2["val_loss"],   linewidth=2, linestyle="--", label="Val MAE (Center)")
ax.set_title("Sentinel-2 · MAE (Center)")
ax.grid(True, alpha=0.3)
ax.legend()

# (1,3) Sentinel-2 · SAM (Center)
ax = axes[0, 2]
ax.plot(s2["epoch"], s2["train_sam"], linewidth=2, label="Train SAM (Center)")
ax.plot(s2["epoch"], s2["val_sam"],   linewidth=2, linestyle="--", label="Val SAM (Center)")
ax.set_title("Sentinel-2 · SAM (Center)")
ax.grid(True, alpha=0.3)
ax.legend()

# ---- Row 2: Surroundings ----
# (2,1) Sentinel-1 · MAE (Surroundings)
ax = axes[1, 0]
ax.plot(s1["epoch"], s1["train_mae_surroundings"], linewidth=2, label="Train MAE (Surroundings)")
ax.plot(s1["epoch"], s1["mae_surroundings"],       linewidth=2, linestyle="--", label="Val MAE (Surroundings)")
ax.set_title("Sentinel-1 · MAE (Surroundings)")
ax.set_xlabel("Epoch")
ax.grid(True, alpha=0.3)
ax.legend()

# (2,2) Sentinel-2 · MAE (Surroundings)
ax = axes[1, 1]
ax.plot(s2["epoch"], s2["train_mae_surroundings"], linewidth=2, label="Train MAE (Surroundings)")
ax.plot(s2["epoch"], s2["mae_surroundings"],       linewidth=2, linestyle="--", label="Val MAE (Surroundings)")
ax.set_title("Sentinel-2 · MAE (Surroundings)")
ax.set_xlabel("Epoch")
ax.grid(True, alpha=0.3)
ax.legend()

# (2,3) Sentinel-2 · SSIM (Surroundings)
ax = axes[1, 2]
ax.plot(s2["epoch"], s2["train_ssim"], linewidth=2, label="Train SSIM Loss (Surroundings)")
ax.plot(s2["epoch"], s2["val_ssim"],   linewidth=2, linestyle="--", label="Val SSIM Loss (Surroundings)")
ax.set_title("Sentinel-2 · SSIM (Surroundings)")
ax.set_xlabel("Epoch")
ax.grid(True, alpha=0.3)
ax.legend()

# ---- Harmonize y-lims for MAE panels (optional) ----
# Row 1: MAE(center) = (0,0) & (0,1)
ymins = [axes[0,0].get_ylim()[0], axes[0,1].get_ylim()[0]]
ymaxs = [axes[0,0].get_ylim()[1], axes[0,1].get_ylim()[1]]
axes[0,0].set_ylim(min(ymins), max(ymaxs))
axes[0,1].set_ylim(min(ymins), max(ymaxs))

# Row 2: MAE(surroundings) = (1,0) & (1,1)
ymins = [axes[1,0].get_ylim()[0], axes[1,1].get_ylim()[0]]
ymaxs = [axes[1,0].get_ylim()[1], axes[1,1].get_ylim()[1]]
axes[1,0].set_ylim(min(ymins), max(ymaxs))
axes[1,1].set_ylim(min(ymins), max(ymaxs))

# ---- Keep distance from x=0 on every panel ----
for ax in axes.flat:
    ax.set_xmargin(0.03)      # ~3% horizontal padding
    ax.autoscale(enable=True, axis='x', tight=False)

fig.tight_layout(rect=[0, 0, 1, 0.97])

# Save
out_png = "./plots/s1_s2_losses_grid.png"
out_pdf = "./plots/s1_s2_losses_grid.pdf"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}, {out_pdf}")

plt.show()
