#!/usr/bin/env python3
import re
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple, List, Set
from sites import sites_dict

# ------------------------
# Paths / config
# ------------------------
ROOT_DIR = Path("/net/data/Fluxnet/")
IN_DIR   = Path("/net/data_ssd/deepfeatures/sciencecubes_processed")
OUT_DIR  = Path("/net/data_ssd/deepfeatures/sciencecubes_processed")

# Process these cubes; by default use all known keys from sites_dict
# (Replace with a manual subset if you like)

CUBE_IDS = [ "003", "008", "005", "028", "027", "036", "011", "021", "022"]

# Feature source to use for GPP dataset creation.
# "ucm_flux" reads the original processed cube with flux-aware UCM filling.
# "linear" reads the output written by GPP_modelling/linear.py.
FEATURE_SOURCE = "linear"

FEATURE_SOURCE_CONFIG = {
    "ucm_flux": {
        "path_template": "s1_s2_{cid}_mean_ucm_flux.zarr",
        "var_name": "feature_mean_ucm",
        "output_tag": "ucm_flux",
    },
    "linear": {
        "path_template": "s1_s2_{cid}_mean_linear.zarr",
        "var_name": "feature_mean_linear",
        "output_tag": "linear",
    },
}
WINDOW   = 90
OVERLAP  = 80  # erlaubte Überlappung in Tagen
assert 0 <= OVERLAP < WINDOW, "OVERLAP muss in [0, WINDOW) liegen"
STRIDE = WINDOW - OVERLAP

# Year split (by window end_date)
TRAIN_YEARS = {2017, 2018, 2019}
VAL_YEARS   = {2020}
YEARS_OF_INTEREST = TRAIN_YEARS | VAL_YEARS  # Will trim cubes to these years

# --- radiation handling ---
# Options:
#   "include" -> keep radiation feature and standardize it with RAD_MEAN/RAD_STD
#   "exclude" -> drop radiation feature entirely from inputs
RADIATION_MODE = "exclude"   # change to "exclude" to drop radiation

# --- global standardization constants (used if RADIATION_MODE == "include") ---
RAD_MEAN = 28.8545
RAD_STD  = 6.8393

# --- target (GPP) standardization ---
GPP_MEAN = 4.042
GPP_STD  = 4.386

# QC threshold (%). Set to 70.0 for ≥70% valid, 0.0 to accept all.
QC_THRESH = 70.0


# ------------------------
# Helper functions
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
    """Open (feature,time) DataArray + radiation index."""
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
    da = _safe(ds[var_name]).sortby("time")
    ridx = ds[var_name].attrs.get("radiation_feature_index", None)
    return da, int(ridx) if ridx is not None else None

def _parse_date_col(df: pd.DataFrame) -> pd.Series:
    cols = [c.strip().lstrip("\ufeff") for c in df.columns]
    candidates = [c for c in cols if c.upper().startswith("TIMESTAMP")]
    if not candidates:
        raise ValueError("No TIMESTAMP column")
    col = candidates[0]
    s = df[col].astype(str)

    m8  = s.str.match(r"^\d{8}$")
    m12 = s.str.match(r"^\d{12}$")
    if m8.any():
        s.loc[m8]  = pd.to_datetime(s[m8],  format="%Y%m%d",      errors="coerce").dt.strftime("%Y-%m-%d")
    if m12.any():
        s.loc[m12] = pd.to_datetime(s[m12], format="%Y%m%d%H%M", errors="coerce").dt.strftime("%Y-%m-%d")

    dt = pd.to_datetime(s, errors="coerce", utc=False)
    return dt.dt.normalize()


def _choose_gpp_column(cols: List[str]) -> Optional[str]:
    pref = [c for c in cols if c.upper() == "GPP_NT_VUT_REF"]
    if pref:
        return pref[0]
    alt = [c for c in cols if re.match(r"(?i)^GPP($|[_])", c)]
    return alt[0] if alt else None

def _load_fluxnet_daily_gpp(site: str) -> pd.Series:
    """
    Load daily GPP (µmol CO₂ m⁻² s⁻¹) for a given site.
    Includes only days with QC ≥ QC_THRESH (interpreted as percent if >1).
    Invalid or missing values are skipped.
    """
    files: List[Path] = []
    ww_dir = ROOT_DIR / "FLUXNET2020-ICOS-WarmWinter"
    if ww_dir.exists():
        files += list(ww_dir.glob(f"FLX_*{site}*_FLUXNET2015_FULLSET_DD_*_beta-3.csv"))
    if not files:
        raise FileNotFoundError(f"No FLUXNET WarmWinter DD files for site {site}")

    parts: List[pd.Series] = []

    for f in files:
        #df = pd.read_csv(f, low_memory=False)
        df = pd.read_csv(f, low_memory=False, encoding="utf-8-sig")
        dt = _parse_date_col(df)

        gcol = _choose_gpp_column(df.columns.tolist())
        if not gcol or gcol not in df.columns:
            continue

        qc_candidates = [c for c in df.columns if "QC" in c.upper()]
        if not qc_candidates:
            continue
        qc_col = qc_candidates[0]

        gpp = pd.Series(df[gcol].astype(float).values, index=dt)
        qc  = pd.Series(df[qc_col].astype(float).values, index=dt)

        # interpret QC (0..1) as percent
        if qc.max(skipna=True) <= 1.1:
            qc = qc * 100.0

        valid_mask = qc >= QC_THRESH
        valid_days = gpp[valid_mask & gpp.notna() & np.isfinite(gpp)]

        if not valid_days.empty:
            parts.append(valid_days)

    if not parts:
        raise ValueError(f"No valid GPP data (QC ≥ {QC_THRESH}%) for {site}")

    df_stack = pd.concat(parts, axis=1)
    gpp = df_stack.bfill(axis=1).iloc[:, 0]  # first valid per day
    gpp = gpp.sort_index()
    gpp.name = "GPP"

    return gpp

def _standardize_radiation(da_ft: xr.DataArray, ridx: Optional[int]) -> xr.DataArray:
    """Standardize the radiation feature (if present)."""
    if ridx is None:
        return da_ft
    da = da_ft.copy()
    rad = da.isel(feature=ridx)
    da.loc[dict(feature=ridx)] = (rad - RAD_MEAN) / (RAD_STD or 1.0)
    return da

def _apply_radiation_mode(da_ft: xr.DataArray, ridx: Optional[int]) -> xr.DataArray:
    """Include (with standardization) or exclude the radiation feature."""
    mode = RADIATION_MODE.lower()
    if mode not in {"include", "exclude"}:
        raise ValueError(f"Invalid RADIATION_MODE: {RADIATION_MODE}")

    if ridx is None:
        # If we don't know the radiation index, just return as-is
        return da_ft

    if mode == "include":
        return _standardize_radiation(da_ft, ridx)

    # mode == "exclude"
    try:
        return da_ft.drop_isel(feature=[ridx])
    except Exception:
        sel = np.ones(da_ft.sizes["feature"], dtype=bool)
        sel[ridx] = False
        return da_ft.isel(feature=np.where(sel)[0])

def _trim_to_years(da_ft: xr.DataArray, years: Set[int]) -> xr.DataArray:
    """Trim DataArray to the provided set of years."""
    if not years:
        return da_ft.isel(time=slice(0, 0))
    tyears = pd.to_datetime(da_ft.time.values).year
    mask = np.isin(tyears, sorted(years))
    if not mask.any():
        return da_ft.isel(time=slice(0, 0))
    return da_ft.isel(time=np.where(mask)[0])

def _make_windows(da_ft: xr.DataArray,
                  gpp: pd.Series,
                  cid: str,
                  site: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build WINDOW-day windows with STRIDE and standardized GPP target on last day."""
    times = pd.to_datetime(da_ft["time"].values).normalize()
    F = da_ft.sizes["feature"]

    Xs, ys, meta = [], [], []
    for end_idx in range(WINDOW - 1, len(times), STRIDE):
        start_idx = end_idx - (WINDOW - 1)
        win_times = times[start_idx:end_idx + 1]
        end_day = win_times[-1]

        tgt = gpp.get(end_day, np.nan)
        if pd.isna(tgt):
            continue

        tgt_std = (tgt - GPP_MEAN) / (GPP_STD or 1.0)

        win = da_ft.isel(time=slice(start_idx, end_idx + 1))
        arr = np.asarray(win.values, np.float32)
        if not np.isfinite(arr).all():
            continue

        Xs.append(arr)
        ys.append(float(tgt_std))
        meta.append({
            "cube_id": cid,
            "site": site,
            "end_date": str(end_day.date())
        })

    if not Xs:
        return (np.empty((0, F, WINDOW), np.float32),
                np.empty((0,), np.float32),
                pd.DataFrame(columns=["cube_id","site","end_date"]))

    X = np.stack(Xs, axis=0)
    y = np.array(ys, np.float32)
    return X, y, pd.DataFrame(meta)

# ------------------------
# Main
# ------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect train/val separately
    X_tr_list, y_tr_list, meta_tr_list = [], [], []
    X_va_list, y_va_list, meta_va_list = [], [], []

    for cid in CUBE_IDS:
        site = sites_dict.get(cid, [None])[0]
        if site is None:
            print(f"⚠️  Missing site mapping for cube {cid}")
            continue

        flux_years = detect_flux_years_for_site(site, ROOT_DIR)
        # Restrict to years relevant for this split and actually present for the site
        active_years = sorted((flux_years & YEARS_OF_INTEREST))
        if not active_years:
            print(f"→ {cid} ({site}): no relevant years {sorted(YEARS_OF_INTEREST)} — skip")
            continue

        try:
            da_ft, ridx = _open_cube_da(cid)
        except Exception as e:
            print(f"⚠️  Skip {cid}: {e}")
            continue

        # Trim to active years (2017–2020 intersection with what's available)
        da_ft = _trim_to_years(da_ft, set(active_years))
        if da_ft.sizes.get("time", 0) < WINDOW:
            print(f"→ {cid} ({site}): too few days after trim — skip")
            continue

        # include or exclude radiation as requested
        da_ft = _apply_radiation_mode(da_ft, ridx)

        try:
            gpp = _load_fluxnet_daily_gpp(site)
        except Exception as e:
            print(f"⚠️  GPP load failed for {site}: {e}")
            continue
        # Keep only days in active years
        gpp = gpp[gpp.index.year.isin(active_years)]

        X, y, meta = _make_windows(da_ft, gpp, cid, site)
        if len(y) == 0:
            print(f"→ {cid} ({site}): 0 samples after windowing — skip")
            continue

        # Split by end_date year
        end_years = pd.to_datetime(meta["end_date"]).dt.year.values
        tr_mask = np.isin(end_years, list(TRAIN_YEARS))
        va_mask = np.isin(end_years, list(VAL_YEARS))

        n_tr = int(tr_mask.sum())
        n_va = int(va_mask.sum())

        if n_tr:
            X_tr_list.append(X[tr_mask])
            y_tr_list.append(y[tr_mask])
            meta_tr_list.append(meta.loc[tr_mask].reset_index(drop=True))
        if n_va:
            X_va_list.append(X[va_mask])
            y_va_list.append(y[va_mask])
            meta_va_list.append(meta.loc[va_mask].reset_index(drop=True))

        print(f"→ {cid} ({site}): total={len(y)}, train={n_tr} (years {sorted(TRAIN_YEARS)}), "
              f"val={n_va} (years {sorted(VAL_YEARS)}), stride={STRIDE}, overlap={OVERLAP}, "
              f"features={da_ft.sizes['feature']} (mode={RADIATION_MODE})")

    # =======================
    # Save outputs
    # =======================
    rad_tag = "rad" if RADIATION_MODE.lower() == "include" else "noRad"
    source_tag = FEATURE_SOURCE_CONFIG[FEATURE_SOURCE]["output_tag"]
    years_tag_tr = f"{min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}"
    years_tag_va = "-".join(str(y) for y in sorted(VAL_YEARS))

    # --- Train ---
    if X_tr_list:
        X_tr = np.concatenate(X_tr_list, axis=0)
        y_tr = np.concatenate(y_tr_list, axis=0)
        meta_tr = pd.concat(meta_tr_list, ignore_index=True)

        npz_tr = OUT_DIR / f"gpp_{WINDOW}day_samples_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}_years{years_tag_tr}_{source_tag}_{rad_tag}_gppstd_train.npz"
        csv_tr = OUT_DIR / f"gpp_{WINDOW}day_samples_meta_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}_years{years_tag_tr}_{source_tag}_{rad_tag}_train.csv"

        np.savez_compressed(npz_tr, X=X_tr, y=y_tr)
        meta_tr.to_csv(csv_tr, index=False)

        print(f"\n✅ Saved TRAIN ({years_tag_tr}, {RADIATION_MODE}):")
        print(f"   → {npz_tr}")
        print(f"   → {csv_tr}")
        print(f"   Shapes: X={X_tr.shape}, y={y_tr.shape}")
    else:
        print("\n⚠️  No TRAIN samples produced.")

    # --- Val ---
    if X_va_list:
        X_va = np.concatenate(X_va_list, axis=0)
        y_va = np.concatenate(y_va_list, axis=0)
        meta_va = pd.concat(meta_va_list, ignore_index=True)

        npz_va = OUT_DIR / f"gpp_{WINDOW}day_samples_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}_years{years_tag_va}_{source_tag}_{rad_tag}_gppstd_val.npz"
        csv_va = OUT_DIR / f"gpp_{WINDOW}day_samples_meta_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}_years{years_tag_va}_{source_tag}_{rad_tag}_val.csv"

        np.savez_compressed(npz_va, X=X_va, y=y_va)
        meta_va.to_csv(csv_va, index=False)

        print(f"\n✅ Saved VAL ({years_tag_va}, {RADIATION_MODE}):")
        print(f"   → {npz_va}")
        print(f"   → {csv_va}")
        print(f"   Shapes: X={X_va.shape}, y={y_va.shape}")
    else:
        print("\n⚠️  No VAL samples produced.")

if __name__ == "__main__":
    main()
