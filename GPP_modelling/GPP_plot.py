#!/usr/bin/env python3
import re
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Optional, Tuple, List, Set, Dict
import torch
import torch.nn as nn

# ------------------------
# Project imports
# ------------------------

from model import GPPTemporalTransformer
from sites import sites_dict

# ------------------------
# User config (7-feature model ONLY)
# ------------------------
CKPT_PATH_7 = "/scratch/jpeters/CA_MM_Embeddings/GPP_modelling/grid_logs_linear_noRad_2017-2019_to_2020/bs6_dm96_h8_L3_ff1024_do0p05_last_lr0p0001_wd1e-06_wu300_rp7_rf0p1/checkpoints/best-epoch=55-val_loss=0.4103.ckpt"

ROOT_DIR = Path("/net/data/Fluxnet/")
IN_DIR   = Path("/net/data_ssd/deepfeatures/sciencecubes_processed")
OUT_DIR  = Path("./gpp_compare_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUBE_IDS = [ "003", "027", "022" ]

FEATURE_SOURCE = "linear"
FEATURE_SOURCE_CONFIG = {
    "ucm_flux": {
        "path_template": "s1_s2_{cid}_v_mean_ucm_flux.zarr",
        "var_name": "feature_mean_ucm",
    },
    "linear": {
        "path_template": "s1_s2_{cid}_v_mean_linear.zarr",
        "var_name": "feature_mean_linear",
    },
}

TRAIN_YEARS = {2017, 2018, 2019}
VAL_YEARS   = {2020}

BIOME_TAGS = {
    "CZ-Lnz": "DBF",
    "IT-Lav": "ENF",
    "CH-Fru": "GRA",
    "IT-MBo": "GRA",
    "CH-Cha": "GRA",
    "IT-Ren": "ENF",
    "BE-Bra": "ENF",
}

WINDOW   = 90
OVERLAP  = 89
assert 0 <= OVERLAP < WINDOW, "OVERLAP must be in [0, WINDOW)"
STRIDE = WINDOW - OVERLAP

# Build windows WITH radiation standardized; we’ll drop it for the 7-feat model.
RADIATION_FOR_WINDOWS = "exclude"

# --- constants (must match training) ---
RAD_MEAN = 28.8545
RAD_STD  = 6.8393
GPP_MEAN = 4.042
GPP_STD  = 4.386

QC_THRESH = 0.0  # use 70.0 if you want QC≥70%

# ------------------------
# Helpers
# ------------------------
def _site_in_filename(site: str, name: str) -> bool:
    return site in name

def detect_flux_years_for_site(site: str, root: Path) -> Set[int]:
    years: Set[int] = set()
    ww_dir = root / "FLUXNET2020-ICOS-WarmWinter"
    if ww_dir.exists():
        for p in ww_dir.glob("FLX_*_FLUXNET2015_FULLSET_DD_*_beta-3.csv"):
            if _site_in_filename(site, p.name):
                years.update({2017, 2018, 2019, 2020})
                break
    return {y for y in years if 2017 <= y <= 2020}

def _safe(da: xr.DataArray) -> xr.DataArray:
    return xr.where(np.isfinite(da), da, np.nan)

def _open_cube_da(cid: str) -> Tuple[xr.DataArray, Optional[int]]:
    if FEATURE_SOURCE not in FEATURE_SOURCE_CONFIG:
        raise ValueError(f"Unknown FEATURE_SOURCE: {FEATURE_SOURCE}")
    cfg = FEATURE_SOURCE_CONFIG[FEATURE_SOURCE]
    z = IN_DIR / cfg["path_template"].format(cid=cid)
    if not z.exists():
        raise FileNotFoundError(z)
    ds = xr.open_zarr(z, consolidated=True)
    var_name = cfg["var_name"]
    if var_name not in ds:
        raise KeyError(f"{var_name} not in {z}")
    da = _safe(ds[var_name]).sortby("time")  # (feature, time)
    ridx = ds[var_name].attrs.get("radiation_feature_index", None)
    return da, int(ridx) if ridx is not None else None

def _parse_date_col(df: pd.DataFrame) -> pd.Series:
    candidates = [c for c in df.columns if c.upper().startswith("TIMESTAMP")]
    if not candidates:
        raise ValueError("No TIMESTAMP column")
    col = candidates[0]
    s = df[col].astype(str)
    m8 = s.str.match(r"^\d{8}$")
    if m8.any():
        s.loc[m8] = pd.to_datetime(s[m8], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    m12 = s.str.match(r"^\d{12}$")
    if m12.any():
        s.loc[m12] = pd.to_datetime(s[m12], format="%Y%m%d%H%M", errors="coerce").dt.strftime("%Y-%m-%d")
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    return dt.dt.normalize()

def _choose_gpp_column(cols: List[str]) -> Optional[str]:
    pref = [c for c in cols if c.upper() == "GPP_NT_VUT_REF"]
    if pref: return pref[0]
    alt = [c for c in cols if re.match(r"(?i)^GPP($|[_])", c)]
    return alt[0] if alt else None

def _load_fluxnet_daily_gpp(site: str, qc_thresh: float = QC_THRESH) -> pd.Series:
    files: List[Path] = []
    ww_dir = ROOT_DIR / "FLUXNET2020-ICOS-WarmWinter"
    if ww_dir.exists():
        files += list(ww_dir.glob(f"FLX_*{site}*_FLUXNET2015_FULLSET_DD_*_beta-3.csv"))
    if not files:
        raise FileNotFoundError(f"No FLUXNET DD files for site {site}")

    parts: List[pd.Series] = []

    for f in files:
        df = pd.read_csv(f, low_memory=False)
        dt = _parse_date_col(df)
        gcol = _choose_gpp_column(df.columns.tolist())
        if not gcol or gcol not in df.columns:
            continue
        qc_candidates = [c for c in df.columns if "QC" in c.upper()]
        if not qc_candidates:
            continue
        qc_col = qc_candidates[0]

        gpp = pd.Series(pd.to_numeric(df[gcol], errors="coerce").values, index=dt)
        qc  = pd.Series(pd.to_numeric(df[qc_col], errors="coerce").values, index=dt)

        if qc.max(skipna=True) <= 1.1:
            qc = qc * 100.0

        valid_mask = qc >= qc_thresh
        valid_days = gpp[valid_mask & gpp.notna() & np.isfinite(gpp)]
        if not valid_days.empty:
            parts.append(valid_days)

    if not parts:
        raise ValueError(f"No valid GPP data (QC ≥ {qc_thresh}%) for {site}")

    df_stack = pd.concat(parts, axis=1)
    gpp = df_stack.bfill(axis=1).iloc[:, 0]
    gpp = gpp.sort_index()
    gpp.name = "GPP"
    return gpp

def _standardize_radiation(da_ft: xr.DataArray, ridx: Optional[int]) -> xr.DataArray:
    if ridx is None:
        return da_ft
    da = da_ft.copy()
    rad = da.isel(feature=ridx)
    da.loc[dict(feature=ridx)] = (rad - RAD_MEAN) / (RAD_STD or 1.0)
    return da

def _apply_radiation_mode(da_ft: xr.DataArray, ridx: Optional[int], mode: str) -> xr.DataArray:
    mode = mode.lower()
    if mode not in {"include", "exclude"}:
        raise ValueError(f"Invalid radiation mode: {mode}")
    if ridx is None:
        return da_ft
    if mode == "include":
        return _standardize_radiation(da_ft, ridx)
    # exclude -> drop radiation feature
    try:
        return da_ft.drop_isel(feature=[ridx])
    except Exception:
        sel = np.ones(da_ft.sizes["feature"], dtype=bool)
        sel[ridx] = False
        return da_ft.isel(feature=np.where(sel)[0])

def _trim_to_flux_years(da_ft: xr.DataArray, years: List[int]) -> xr.DataArray:
    if not years:
        return da_ft.isel(time=slice(0, 0))
    mask = np.isin(pd.to_datetime(da_ft.time.values).year, years)
    if not mask.any():
        return da_ft.isel(time=slice(0, 0))
    return da_ft.isel(time=np.where(mask)[0])

def _make_windows(da_ft: xr.DataArray,
                  gpp: pd.Series,
                  cid: str,
                  site: str) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    times = pd.to_datetime(da_ft["time"].values).normalize()
    F = da_ft.sizes["feature"]
    Xs, ys, ts = [], [], []
    for end_idx in range(WINDOW - 1, len(times), STRIDE):
        start_idx = end_idx - (WINDOW - 1)
        end_day = times[end_idx]
        tgt = gpp.get(end_day, np.nan)
        if pd.isna(tgt):
            continue
        tgt_std = (tgt - GPP_MEAN) / (GPP_STD or 1.0)
        win = da_ft.isel(time=slice(start_idx, end_idx + 1))
        arr = np.asarray(win.values, np.float32)  # (F, T)
        if not np.isfinite(arr).all():
            continue
        Xs.append(arr.T)  # (T, F)
        ys.append(float(tgt_std))
        ts.append(pd.Timestamp(end_day))
    if not Xs:
        return (np.empty((0, WINDOW, F), np.float32),
                np.empty((0,), np.float32),
                [])
    X = np.stack(Xs, axis=0)  # (N, T, F)
    y = np.array(ys, np.float32)
    return X, y, ts

# ------------------------
# Metrics
# ------------------------
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    r = rmse(a, b)
    denom = float(np.max(b) - np.min(b)) if len(b) > 0 else np.nan
    return r / denom if denom > 0 else np.nan

def overall_nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    mask = np.isfinite(pred) & np.isfinite(true)
    if not mask.any():
        return np.nan
    pred = pred[mask]; true = true[mask]
    return nrmse(pred, true)

# ------------------------
# Model loading + alignment (7 features)
# ------------------------
def _infer_model_feature_count(model: nn.Module) -> int:
    nf = getattr(getattr(model, "hparams", object()), "num_features", None)
    if isinstance(nf, (int, np.integer)):
        return int(nf)
    lin = getattr(model, "input_proj", None)
    if isinstance(lin, nn.Linear):
        return int(lin.in_features)
    raise RuntimeError("Cannot infer model's expected num_features.")

def load_model(ckpt_path: str, device: torch.device) -> Tuple[GPPTemporalTransformer, int, str]:
    model = GPPTemporalTransformer.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval().to(device)
    f_model = _infer_model_feature_count(model)
    ckpt_label = Path(ckpt_path).name
    print(f"[model] {ckpt_label} expects num_features = {f_model}")
    time_first = getattr(model.hparams, "time_first", True)
    assert time_first, "Model was trained with time_first=True; expected (B,T,F) inputs."
    return model, f_model, ckpt_label

def align_X_to_model(X: np.ndarray, f_model: int, ridx: Optional[int]) -> np.ndarray:
    f_data = X.shape[-1]
    if f_data == f_model:
        return X
    if f_data == f_model + 1:
        drop_idx = ridx if ridx is not None else (f_data - 1)
        keep = [i for i in range(f_data) if i != drop_idx]
        return X[..., keep]
    if f_data + 1 == f_model:
        insert_idx = ridx if ridx is not None else f_data
        zeros = np.zeros((*X.shape[:-1], 1), dtype=X.dtype)
        return np.concatenate([X[..., :insert_idx], zeros, X[..., insert_idx:]], axis=-1)
    raise ValueError(f"Cannot align features: data has {f_data}, model expects {f_model}.")

# ------------------------
# Plotting (single-model)
# ------------------------
def plot_per_cube(dates: List[pd.Timestamp],
                  y_true: np.ndarray,
                  y_pred7: np.ndarray,
                  cid: str,
                  site: str,
                  out_dir: Path,
                  label7: str):
    nrmse7 = nrmse(y_pred7, y_true)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(dates, y_true, color="lightgray", s=15, label="True GPP (daily)", zorder=2)
    ax.plot(dates, y_pred7, color="#1f77b4", linewidth=1.4, label=label7, zorder=3)

    ax.text(0.02, 0.96,
            f"{label7}: NRMSE={nrmse7:.3f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"))

    ax.set_title(f"{site}", fontsize=12, weight="semibold")
    ax.set_xlabel("Date")
    ax.set_ylabel("GPP")
    ax.grid(True, linewidth=0.5, alpha=0.5)

    out_png = out_dir / f"gpp_compare_7feat_{cid}_{site.replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"   📈 saved: {out_png}")

    return {
        "cid": cid, "site": site,
        "dates": dates, "y_true": y_true,
        "y7": y_pred7,
        "nrmse7": nrmse7
    }

def plot_combined_7feat(packs: List[Dict],
                        out_dir: Path,
                        label7: str,
                        overall7: float):
    n = len(packs)
    cols, rows = 1, n
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 2.4 * rows), squeeze=False)
    axes = axes.flatten()

    for i, p in enumerate(packs):
        ax = axes[i]
        ax.scatter(p["dates"], p["y_true"], color="lightgray", s=10, zorder=2)
        ax.plot(p["dates"], p["y7"], color="#1f77b4", linewidth=1.2, zorder=3)

        biome = BIOME_TAGS.get(p["site"], None)
        biome_str = f" ({biome})" if biome else ""
        ax.set_title(f"{p['site']}{biome_str}", fontsize=11, weight="semibold")

        ax.set_ylabel("GPP", fontsize=11)
        ax.set_xlabel("")
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.label_outer()
        if i < n - 1:
            ax.set_xticklabels([])

    legend_elements = [
        Line2D([0], [0], color="lightgray", lw=0, marker="o", markersize=6, label="True GPP (daily)"),
        Line2D([0], [0], color="#1f77b4", lw=1.6, label=label7),
    ]
    fig.legend(handles=legend_elements,
               loc="lower center", bbox_to_anchor=(0.5, -0.01),
               frameon=False, fontsize=11, ncol=2,
               columnspacing=1.2, handlelength=2.2)

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_png = out_dir / "gpp_compare_combined_7feat_bottomlegend.png"
    out_pdf = out_dir / "gpp_compare_combined_7feat_bottomlegend.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"   🖼️ combined saved: {out_png}")
    print(f"   🖨️ combined saved: {out_pdf}")

# ------------------------
# Main (7-feature flow)
# ------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")

    split_acc = {
        "train_true": [], "train_y7": [],
        "val_true":   [], "val_y7":   [],
    }

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model7, f7, label7 = load_model(CKPT_PATH_7, device)

    packs: List[Dict] = []
    all_true, all_y7 = [], []

    for cid in CUBE_IDS:
        site = sites_dict.get(cid, [None])[0]
        if site is None:
            print(f"⚠️ Missing site mapping for cube {cid} — skip"); continue

        flux_years = sorted(detect_flux_years_for_site(site, ROOT_DIR))
        if not flux_years:
            print(f"→ {cid} ({site}): no flux years — skip"); continue

        try:
            da_ft, ridx = _open_cube_da(cid)
        except Exception as e:
            print(f"⚠️ Skip {cid}: {e}"); continue

        da_ft = _trim_to_flux_years(da_ft, flux_years)
        if da_ft.sizes.get("time", 0) < WINDOW:
            print(f"→ {cid} ({site}): too few days after trim — skip"); continue
        da_ft = _apply_radiation_mode(da_ft, ridx, RADIATION_FOR_WINDOWS)

        try:
            gpp_series = _load_fluxnet_daily_gpp(site, qc_thresh=QC_THRESH)
        except Exception as e:
            print(f"⚠️  GPP load failed for {site}: {e}"); continue
        gpp_series = gpp_series[gpp_series.index.year.isin(flux_years)]

        X_base, y_std, ts = _make_windows(da_ft, gpp_series, cid, site)
        if X_base.shape[0] == 0:
            print(f"→ {cid} ({site}): no valid windows — skip"); continue

        # align and predict for 7 features
        X7 = align_X_to_model(X_base, f_model=f7, ridx=ridx)
        with torch.no_grad():
            Xt7 = torch.tensor(X7, dtype=torch.float32, device=device)
            y7_std = model7(Xt7).detach().cpu().numpy()

        y7 = y7_std * GPP_STD + GPP_MEAN

        y_true = np.array([gpp_series.get(pd.Timestamp(t).normalize(), np.nan) for t in ts], dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y7)
        dates = [t for t, m in zip(ts, mask) if m]
        y_true = y_true[mask]; y7 = y7[mask]

        years = np.array([d.year for d in dates])
        train_mask = np.isin(years, list(TRAIN_YEARS))
        val_mask   = np.isin(years, list(VAL_YEARS))

        if train_mask.any():
            tr_nrmse7 = nrmse(y7[train_mask], y_true[train_mask])
            print(f"   ↳ Train 2017–2019 NRMSE — 7feat: {tr_nrmse7:.3f}")

        if val_mask.any():
            va_nrmse7 = nrmse(y7[val_mask], y_true[val_mask])
            print(f"   ↳ Val 2020 NRMSE       — 7feat: {va_nrmse7:.3f}")

        if train_mask.any():
            split_acc["train_true"].append(y_true[train_mask])
            split_acc["train_y7"].append(y7[train_mask])
        if val_mask.any():
            split_acc["val_true"].append(y_true[val_mask])
            split_acc["val_y7"].append(y7[val_mask])

        if len(y_true) == 0:
            print(f"→ {cid} ({site}): no comparable points after masking — skip"); continue

        pack = plot_per_cube(
            dates=dates,
            y_true=y_true,
            y_pred7=y7,
            cid=cid,
            site=site,
            out_dir=OUT_DIR,
            label7="prediction from 7 features",
        )
        print(f"[{cid} {site}] NRMSE 7 features (no radiation): {pack['nrmse7']:.3f}")

        out_csv = OUT_DIR / f"gpp_compare_cube_{cid}_7feat.csv"
        df_out = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "year": [d.year for d in dates],
            "split": np.where(np.isin([d.year for d in dates], list(TRAIN_YEARS)), "train",
                              np.where(np.isin([d.year for d in dates], list(VAL_YEARS)), "val", "other")),
            "gpp_true": y_true,
            "gpp_pred_7feat": y7,
        })
        df_out = df_out[df_out["split"].isin(["train","val"])]
        df_out.to_csv(out_csv, index=False)
        print(f"   💾 saved: {out_csv}")

        packs.append(pack)
        all_true.append(y_true); all_y7.append(y7)

    if packs:
        all_true = np.concatenate(all_true, axis=0)
        all_y7   = np.concatenate(all_y7,   axis=0)
        overall7 = overall_nrmse(all_y7, all_true)
        plot_combined_7feat(packs, OUT_DIR, label7="prediction from 7 features", overall7=overall7)
        print(f"Overall NRMSE — 7 features (no radiation): {overall7:.4f}")
    else:
        print("No plots to combine.")

    # ------------------------
    # Global split-wise metrics (7-feature only)
    # ------------------------
    def _concat(lst):
        return np.concatenate(lst, axis=0) if lst else np.empty((0,), dtype=float)

    tr_true = _concat(split_acc["train_true"])
    tr_y7   = _concat(split_acc["train_y7"])
    va_true = _concat(split_acc["val_true"])
    va_y7   = _concat(split_acc["val_y7"])

    if tr_true.size:
        tr_nrmse7 = overall_nrmse(tr_y7, tr_true)
        tr_rmse7  = rmse(tr_y7, tr_true)
        print(f"\nGLOBAL Train (2017–2019) NRMSE — 7feat: {tr_nrmse7:.4f}")
        print(f"GLOBAL Train (2017–2019) RMSE  — 7feat: {tr_rmse7:.4f}")
    else:
        tr_nrmse7 = tr_rmse7 = np.nan
        print("\nGLOBAL Train (2017–2019): no data")

    if va_true.size:
        va_nrmse7 = overall_nrmse(va_y7, va_true)
        va_rmse7  = rmse(va_y7, va_true)
        print(f"GLOBAL Val   (2020)      NRMSE — 7feat: {va_nrmse7:.4f}")
        print(f"GLOBAL Val   (2020)      RMSE  — 7feat: {va_rmse7:.4f}")
    else:
        va_nrmse7 = va_rmse7 = np.nan
        print("GLOBAL Val (2020): no data")

    rows = []
    rows.append({"split": "train_2017_2019", "rmse_7": tr_rmse7, "nrmse_7": tr_nrmse7})
    rows.append({"split": "val_2020",        "rmse_7": va_rmse7, "nrmse_7": va_nrmse7})
    pd.DataFrame(rows).to_csv(OUT_DIR / "global_split_metrics_7feat.csv", index=False)

if __name__ == "__main__":
    main()
