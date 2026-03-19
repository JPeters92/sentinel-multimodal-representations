#!/usr/bin/env python3
import math
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from sites import sites_dict
from typing import Optional, Tuple, Union, Set, List
from pathlib import Path
from pysolar.solar import get_altitude
from timezonefinder import TimezoneFinder
from pysolar.radiation import get_radiation_direct
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
from statsmodels.tsa.statespace.structural import UnobservedComponents


warnings.simplefilter("ignore", ValueWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*concentrate_scale.*", category=FutureWarning)

# ------------------------
# Paths
# ------------------------
ROOT_DIR = Path("/net/data/Fluxnet/")
IN_DIR   = Path("/net/data_ssd/deepfeatures/sciencecubes_processed")
OUT_DIR  = Path("/net/data_ssd/deepfeatures/sciencecubes_processed")

ArrayLike = Union[xr.DataArray, xr.Dataset]

# ------------------------
# Gap-aware fallback thresholds (days) — tune if needed
# ------------------------
SHORT_MAX = 20    # ≤ SHORT_MAX: simple interpolation
MID_MAX   = 35    # SHORT_MAX < gap ≤ MID_MAX: residual PCHIP wrt DOY climatology; > MID_MAX: climatology

# ------------------------
# Helpers
# ------------------------
def _site_in_filename(site: str, name: str) -> bool:
    return site in name

def _daily_potential_radiation_MJm2(lat: float, lon: float, day: pd.Timestamp) -> float:
    """Daily clear-sky shortwave (MJ m^-2 day^-1) via 10-min integration of PySolar direct beam."""
    start = dt.datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=dt.timezone.utc)
    end   = start + dt.timedelta(days=1)
    step  = dt.timedelta(minutes=10)

    E_Jm2 = 0.0
    t = start
    while t < end:
        alt = get_altitude(lat, lon, t)  # degrees
        if alt > 0.0:
            I = get_radiation_direct(t, alt)  # W m^-2
            E_Jm2 += float(I) * step.total_seconds()
        t += step
    return E_Jm2 / 1e6  # → MJ m^-2 day^-1

def compute_radiation_series(lat: float, lon: float, dates: pd.DatetimeIndex) -> pd.Series:
    # dates should be daily; we normalize just in case
    days = pd.DatetimeIndex(pd.to_datetime(dates).normalize().unique())
    vals = [ _daily_potential_radiation_MJm2(lat, lon, d) for d in days ]
    s = pd.Series(vals, index=days, dtype="float32")
    # Reindex back to the original daily index (identical here but keeps order)
    return s.reindex(pd.DatetimeIndex(pd.to_datetime(dates).normalize()), method=None).astype("float32")

def detect_flux_years_for_site(site: str, root: Path) -> Set[int]:
    years: Set[int] = set()
    ww_dir = root / "FLUXNET2020-ICOS-WarmWinter"
    if ww_dir.exists():
        for p in ww_dir.glob("FLX_*_FLUXNET2015_FULLSET_DD_*_beta-3.csv"):
            if _site_in_filename(site, p.name):
                years.update({2017, 2018, 2019, 2020})
                break
    for dpat in ["ICOS_2021_I", "ICOS_2022_I", "ICOS_2023_I", "ICOS_2024_I"]:
        d = root / dpat
        if not d.exists():
            continue
        try:
            y = int(dpat.split("_")[1])
        except Exception:
            continue
        for p in d.glob("ICOSETC_*_FLUXNET_DD_01.csv"):
            if _site_in_filename(site, p.name):
                years.add(y)
                break
    return {y for y in years if 2017 <= y <= 2024}

# ------------------------
# DOY climatology helpers
# ------------------------
def _doy_climatology(ts_all: pd.Series) -> pd.Series:
    """Multi-year DOY climatology (median per DOY) with light smoothing."""
    s = ts_all.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.Series(0.0, index=pd.RangeIndex(1, 367))
    df = s.to_frame("v")
    df["doy"] = df.index.dayofyear
    clim = df.groupby("doy")["v"].median()
    # ensure full 1..366, smooth & fill
    clim = clim.reindex(range(1, 367))
    clim = clim.replace([np.inf, -np.inf], np.nan)
    clim = clim.interpolate("pchip").rolling(7, center=True, min_periods=1).mean()
    return clim  # index: 1..366

def _apply_climatology(year: int, clim_doy: pd.Series) -> pd.Series:
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    doy = idx.dayofyear
    vals = clim_doy.reindex(doy).to_numpy()
    if np.isnan(vals).any():
        vals = pd.Series(vals).interpolate("linear", limit_direction="both").to_numpy()
    return pd.Series(vals, index=idx)

def _nan_blocks(x: pd.Series) -> List[Tuple[int, int]]:
    """Return list of (start, end) indices (inclusive) for contiguous NaN blocks."""
    isn = x.isna().to_numpy()
    if not isn.any():
        return []
    starts = np.where(~isn[:-1] & isn[1:])[0] + 1
    ends   = np.where(isn[:-1] & ~isn[1:])[0]
    if isn[0]:  starts = np.r_[0, starts]
    if isn[-1]: ends   = np.r_[ends, len(isn)-1]
    return list(zip(starts, ends))

# ------------------------
# Gap-aware fallback (fast, robust)
# ------------------------
def _gap_aware_fallback(ts: pd.Series, year: int) -> pd.Series:
    """Fallback depending on gap length:
       ≤ SHORT_MAX: time-linear interpolation
       SHORT_MAX < L ≤ MID_MAX: PCHIP on residuals wrt DOY climatology
       > MID_MAX: climatology only (bounds-safe, optional gentle blend)
    """
    full_idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    day = ts.groupby(ts.index.normalize()).mean()
    y_full = day.reindex(full_idx).astype("float32")

    clim_doy = _doy_climatology(ts).astype("float32")
    clim_year = _apply_climatology(year, clim_doy).astype("float32")

    out = y_full.copy()
    blocks = _nan_blocks(y_full)
    if not blocks:
        # sanitize output
        return out.replace([np.inf, -np.inf], np.nan).astype("float32")

    resid = (y_full - clim_year).astype("float32")

    for s, e in blocks:
        L = e - s + 1
        seg_idx = y_full.index[s:e+1]

        if L <= SHORT_MAX:
            tmp = out.interpolate(method="pchip", limit_direction="both").astype("float32")
            out.loc[seg_idx] = tmp.loc[seg_idx].astype("float32")

        elif SHORT_MAX < L <= MID_MAX:
            filled_resid = resid.interpolate(method="pchip", limit_direction="both").astype("float32")
            out.loc[seg_idx] = (clim_year + filled_resid).loc[seg_idx].astype("float32")

        else:
            # long gaps: pure climatology
            out.loc[seg_idx] = clim_year.loc[seg_idx].astype("float32")

            # Optional gentle 3-day blends at edges (bounds-safe)
            blend = 3
            # left edge: blend inside the NaN block starting at s
            for k in range(1, blend + 1):
                i = s - 1 + k
                if 0 <= i < len(out) and pd.isna(y_full.iloc[i]):
                    prev_i = i - 1
                    if prev_i >= 0:
                        w = k / (blend + 1)
                        out.iloc[i] = (1 - w) * out.iloc[prev_i].astype("float32") + w * clim_year.iloc[i].astype("float32")
            # right edge: blend inside the NaN block ending at e
            for k in range(1, blend + 1):
                i = e + 1 - k
                if 0 <= i < len(out) and pd.isna(y_full.iloc[i]):
                    next_i = i + 1
                    if next_i < len(out):
                        w = k / (blend + 1)
                        out.iloc[i] = (1 - w) * out.iloc[next_i] + w * clim_year.iloc[i]

    out = out.interpolate(method="pchip", limit_direction="both").fillna(clim_year)
    return out.replace([np.inf, -np.inf], np.nan).astype("float32")

# ------------------------
# UCM fill with fast seasonal + robust guards
# ------------------------
MIN_OBS = 5  # observations within the year required to try UCM

def _safe_time_fill_for_year(ts: pd.Series, year: int) -> pd.Series:
    """Gap-aware fallback wrapper used by UCM when data are sparse or fit fails."""
    return _gap_aware_fallback(ts, year)

def _is_converged(res) -> bool:
    """Robust convergence check across optimizers/statsmodels versions."""
    if res is None:
        return False
    # Primary, uniform flag
    if getattr(res, "converged", None) is True:
        return True

    # Fallbacks (optimizer-specific)
    ret = getattr(res, "mle_retvals", {}) or {}
    flags = []
    if "converged" in ret:
        flags.append(bool(ret["converged"]))
    if "success" in ret:        # scipy.minimize-based
        flags.append(bool(ret["success"]))
    if "warnflag" in ret:       # scipy.optimize.fmin* (0 means success)
        flags.append(ret["warnflag"] == 0)

    return any(flags) if flags else False

def _fit_ucm_with_retries(mod):
    """Try multiple optimizers; return the first *converged* result, else None."""
    attempts = [
        dict(method="lbfgs",  maxiter=600),
        dict(method="powell", maxiter=900),
        dict(method="nm",     maxiter=900),
    ]
    last_res = None
    for kw in attempts:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)  # capture convergence warnings
            try:
                res = mod.fit(disp=False, **kw)
                last_res = res
                # Treat any ConvergenceWarning as non-converged
                saw_conv_warn = any(isinstance(x.message, ConvergenceWarning) for x in w)
                if _is_converged(res) and not saw_conv_warn:
                    return res
            except Exception:
                continue
    # If you prefer to fall back rather than return a non-converged result:
    return None

def ucm_fill_one_year(ts: pd.Series, year: int) -> pd.Series:
    ts = ts.sort_index().replace([np.inf, -np.inf], np.nan)
    ts_year = ts[(ts.index.year == year)]
    if len(ts_year) == 0:
        return _safe_time_fill_for_year(ts, year)

    # normalize to dates (aggregate duplicates)
    ts_year = ts_year.groupby(ts_year.index.normalize()).mean()
    ts_year.index = pd.DatetimeIndex(ts_year.index)

    if ts_year.notna().sum() < MIN_OBS:
        return _safe_time_fill_for_year(ts, year)

    full_idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    is_leap = (pd.Timestamp(f"{year}-12-31").dayofyear == 366)
    model_idx = full_idx
    if is_leap:
        model_idx = model_idx[~((model_idx.month == 2) & (model_idx.day == 29))]

    y = ts_year.reindex(model_idx).astype(float)

    if pd.isna(y).sum() == len(y) or y.notna().sum() < MIN_OBS:
        return _safe_time_fill_for_year(ts, year)

    # --- NEW: standardize for a well-scaled likelihood ---
    mu = float(np.nanmean(y))
    sd = float(np.nanstd(y))
    if not np.isfinite(sd) or sd == 0.0:
        sd = 1.0
    y_std = (y - mu) / sd

    try:
        mod = UnobservedComponents(
            y_std,
            level="local level",
            freq_seasonal=[{"period": 365, "harmonics": 3}],
            concentrate_scale=True,              # fewer params → easier optimization
        )

        res = _fit_ucm_with_retries(mod)
        if res is None or not _is_converged(res):
            print(f"UCM did not converge → gap-aware fallback year: {year}")
            return _safe_time_fill_for_year(ts, year)


        yhat_std = res.predict()

        # Guard: prediction must be finite
        if not np.isfinite(np.asarray(yhat_std)).all():
            print(f"UCM produced non-finite values → gap-aware fallback year: {year}")
            return _safe_time_fill_for_year(ts, year)

        # invert standardization
        yhat = yhat_std * sd + mu
        print("standard fill")

    except Exception as e:
        print(f"UCM exception ({e}) → gap-aware fallback year: {year}")
        return _safe_time_fill_for_year(ts, year)

    if is_leap:
        full_hat = yhat.reindex(full_idx)
        full_hat.loc[f"{year}-02-29"] = 0.5 * (
            full_hat.loc[f"{year}-02-28"] + full_hat.loc[f"{year}-03-01"]
        )
        return full_hat.sort_index()
    else:
        return yhat


def fill_feature_means_one_year(mean_da: xr.DataArray, year: int) -> xr.DataArray:
    """mean_da: (feature, time) → daily-filled (feature, time) for `year`."""
    features = mean_da["feature"].values
    full_idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    filled_list = []
    for f in features:
        ts = mean_da.sel(feature=f).to_series()
        filled = ucm_fill_one_year(ts, year)
        filled_list.append(xr.DataArray(filled.values.astype("float32"), coords={"time": filled.index}, dims=["time"]))
    out = xr.concat(filled_list, dim="feature").assign_coords(feature=features)
    return out.reindex(time=full_idx)

# ------------------------
# MAIN LOOP
# ------------------------
CUBE_IDS = ["000","001", "003", "004", "005", "006", "007", "008", "009", '011', '012', '013', '016', '017', '018', "019", "020", "021", '022', '024']
CUBE_IDS = ['025', '026', '027',"028", '029', '030', '034', '035', '036', '038', '039', '040', '042', '044', '046']

for cid in CUBE_IDS:
    print(f"\n→ processing cube {cid}")
    in_path  = IN_DIR / f"s1_s2_{cid}.zarr"
    out_path = OUT_DIR / f"s1_s2_{cid}_mean_ucm_flux.zarr"

    ds = xr.open_zarr(in_path, consolidated=True)
    da = ds["features"]  # (feature, time, y, x)

    # 1) spatial mean
    mean_da = da.mean(dim=("y", "x"))

    # 2) restrict to relevant FLUX years
    site_code = sites_dict[cid][0]
    flux_years = set(detect_flux_years_for_site(site_code, ROOT_DIR))
    years_have = set(np.unique(mean_da["time"].dt.year.values).astype(int))
    target_years = sorted(years_have & flux_years)

    print(f"   site={site_code}, flux years={sorted(flux_years) or 'NONE'}")
    print(f"   years in cube={sorted(years_have)}")
    print(f"   → target years={target_years or 'NONE'}")

    if not target_years:
        print("   ⚠️ No overlapping years — skipping.")
        continue

    mean_da = mean_da.sel(time=mean_da.time.where(mean_da.time.dt.year.isin(target_years), drop=True))
    print(f"   restricted mean_da shape: {mean_da.shape}")

    # 3) fill per year and concat
    yearly_filled = [fill_feature_means_one_year(mean_da, y) for y in target_years]
    filled_all = xr.concat(yearly_filled, dim="time").sortby("time")  # (feature, time)

    # --- NEW: append daily potential radiation as 8th feature ---
    # coords from sites_dict: ['SITE', [lat, lon]]
    lat, lon = sites_dict[cid][1]
    dates_all = pd.to_datetime(filled_all["time"].values).normalize()
    rad_series = compute_radiation_series(lat, lon, pd.DatetimeIndex(dates_all))  # MJ m^-2 day^-1

    rad_da = xr.DataArray(
        rad_series.values.astype("float32"),
        coords={"time": filled_all["time"].values},
        dims=["time"],
        name="radiation_potential"
    ).expand_dims("feature").assign_coords(feature=[int(filled_all.sizes["feature"])])

    filled_all = xr.concat([filled_all, rad_da], dim="feature")  # now feature=8
    # ------------------------------------------------------------

    # sanitize any ±inf just in case
    filled_all = xr.where(np.isfinite(filled_all), filled_all, np.nan)

    # --- NaN stats BEFORE writing ---
    n_total = int(np.prod(filled_all.shape))
    n_nan = int(np.isnan(filled_all).sum())
    frac_nan = (n_nan / n_total * 100) if n_total else 0.0
    print(f"   ⚠️ NaN count (incl. radiation): {n_nan:,} / {n_total:,}  ({frac_nan:.2f}%)")

    # 4) write combined Zarr per cube (once)
    ds_out = filled_all.to_dataset(name="feature_mean_ucm")
    # (optional) annotate radiation metadata
    ds_out["feature_mean_ucm"].attrs["radiation_feature_index"] = int(filled_all.sizes["feature"]) - 1
    ds_out["feature_mean_ucm"].attrs["radiation_units"] = "MJ m^-2 day^-1 (clear-sky, PySolar)"

    nfeat = ds_out.sizes["feature"]
    enc = {"feature_mean_ucm": {"chunks": (min(64, nfeat), 180)}}
    ds_out.to_zarr(out_path, mode="w", consolidated=True, encoding=enc)
    print(f"   ✅ saved combined Zarr: {out_path}")
