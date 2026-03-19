import os
import gc
import logging
import time
import torch
import stackstac
import numpy as np
import xarray as xr
import pandas as pd
import planetary_computer as pc
from typing import Tuple, Dict, Optional
from dataset.utils import compute_time_gaps
from pystac_client import Client
from rasterio.enums import Resampling
from dataset.prepare_dataarray import prepare_spectral_data

logger = logging.getLogger(__name__)


def ensure_band(var: xr.DataArray) -> xr.DataArray:
    if "band" in var.dims:
        return var
    if "index" in var.dims:
        return var.rename({"index": "band"})
    raise ValueError(f"No band-like dim in {var.dims}")


def nearest_indices(src_times: np.ndarray, ref_times: np.ndarray) -> np.ndarray:
    """For each time in src_times, return index of the nearest time in ref_times.
        Duplicates are allowed and preserved."""
    ref = np.asarray(ref_times)
    src = np.asarray(src_times)

    order = np.argsort(ref)
    ref_sorted = ref[order]

    pos = np.searchsorted(ref_sorted, src)
    i0 = np.clip(pos - 1, 0, ref_sorted.size - 1)
    i1 = np.clip(pos, 0, ref_sorted.size - 1)

    pick_sorted = np.where(
        np.abs(ref_sorted[i1] - src) < np.abs(ref_sorted[i0] - src),
        i1, i0
    )
    return order[pick_sorted]  # indices into ref_times


def utm_zone_to_epsg(utm: str) -> int:
    zone = int(utm[:-1])
    band = utm[-1].upper()
    return 32600 + zone if band >= 'N' else 32700 + zone


def download_sentinel1_stack(
        bbox_deg,
        start_date,
        end_date,
        collection="sentinel-1-rtc",
        resolution=10,
        utm='35U',
        max_items=5000,
        bands=["vv", "vh"],
        resampling_method=Resampling.bilinear
) -> xr.Dataset:

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    search = catalog.search(
        collections=[collection],
        bbox=bbox_deg,
        datetime=f"{start_date}/{end_date}",
        max_items=max_items
    )

    items = list(search.items())
    if not items:
        raise ValueError("No Sentinel-1 items found for the given search parameters.")

    epsg = utm_zone_to_epsg(utm)
    signed_items = [pc.sign(item) for item in items]

    stacked = stackstac.stack(
        signed_items,
        assets=bands,
        resolution=resolution,
        bounds_latlon=bbox_deg,
        xy_coords='center',
        epsg=epsg,
        resampling=resampling_method,
    )

    #stacked = stacked.transpose("band", "time", "y", "x")

    ds = stacked.to_dataset(name="backscatter")
    ds.attrs = {}
    for var in ds.data_vars:
        ds[var].attrs = {}


    return ds

def match_sentinel1_to_s2_cube(s2: xr.Dataset) -> xr.Dataset:
    """
    Given an S2 cube, download a matching S1 cube with identical x/y coordinates
    and a time range ±7 days around the S2 cube's time extent.
    """
    # Extract spatial and temporal info from s2
    bbox = s2.bbox_wgs84
    s2_x = s2.x.values
    s2_y = s2.y.values

    utm_zone = s2.attrs.get("utm_zone")

    start_time = pd.to_datetime(s2.time.values[0])
    end_time = pd.to_datetime(s2.time.values[-1])
    start_date = (start_time - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (end_time + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    # Download S1 stack
    s1 = download_sentinel1_stack(
        bbox_deg=bbox,
        start_date=start_date,
        end_date=end_date,
        utm=utm_zone
    )


    s1_matched = s1.sel(x=s2_x, y=s2_y, method="nearest")

    # --- Temporal smoothing in LINEAR scale (centered 5, stride=1; edges 3/4) ---
    # If rain spikes are an issue, you can swap .mean() -> .median()
    s1_matched = s1_matched.rolling(time=3, center=True, min_periods=1).mean()

    # --- Transform to dB scale ---
    s1_db = 10 * np.log10(s1_matched)  # avoid log(0)

    # --- Clip to typical S1 range (-30 to +5 dB) ---
    s1_db_clipped = s1_db.clip(min=-30, max=5)

    # --- Normalize to [0,1] ---
    s1_norm = (s1_db_clipped + 30) / 35  # -30→0, +5→1

    return s1_norm


def extract_s1_patches(
    s2_coords: dict,
    s1_array: np.ndarray,      # (bands, T, H, W)
    s1_times: list,             # length T, datetime-like
    s1_x: list,                 # full x coordinate vector of S1 cube
    s1_y: list,                 # full y coordinate vector of S1 cube
    max_time_diff_days: int = 3
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Extract Sentinel-1 patches aligned to selected Sentinel-2 sample times.

    Args:
        s2_coords: dict with keys:
            - 'time': (N, K) timestamps per S2 sample
            - 'x':    (N, P) coordinate values (x) for each S2 patch
            - 'y':    (N, P) coordinate values (y) for each S2 patch
        s1_array: (bands, T, H, W) S1 data cube
        s1_times: list/array of datetime-like, length T
        s1_x: full x coordinate vector of S1 cube (list)
        s1_y: full y coordinate vector of S1 cube (list)
        max_time_diff_days: tolerance in days for S1↔S2 matching

    Returns:
        s1_patches_np:  (N, bands, time, H_sel, W_sel) NaNs filled
        s1_coords:      dict with 'time', 'x', 'y'
        s1_valid_mask_np: bool array, same shape as s1_patches_np (True where valid data)
    """
    # Convert to np.array for fast comparison
    s1_x = np.array(s1_x)
    s1_y = np.array(s1_y)
    s1_times = np.array(s1_times, dtype='datetime64[ns]')

    N = s2_coords["time"].shape[0]
    bands, T, H_full, W_full = s1_array.shape

    patches = []
    coords_time = []
    coords_x = []
    coords_y = []
    valid_masks = []

    for i in range(N):
        # --- Map coords from S2 → indices in S1 ---
        x_idx = np.array([np.where(s1_x == val)[0][0] for val in s2_coords["x"][i]], dtype=int)
        y_idx = np.array([np.where(s1_y == val)[0][0] for val in s2_coords["y"][i]], dtype=int)

        # print(x_idx, y_idx)

        # --- Temporal selection: match each S2 time to closest S1 time ---
        target_times = pd.to_datetime(s2_coords["time"][i])
        target_times = np.array(target_times, dtype='datetime64[ns]')

        matched_indices = []
        matched_times = []

        for t in target_times:
            diffs = np.abs(s1_times - t)
            min_pos = int(np.argmin(diffs))
            min_diff = diffs[min_pos]
            if min_diff <= np.timedelta64(max_time_diff_days, 'D'):
                matched_indices.append(min_pos)
                matched_times.append(s1_times[min_pos])
            else:
                matched_indices.append(None)
                matched_times.append(np.datetime64('NaT'))

        # --- Extract patch ---
        H_sel = len(y_idx)
        W_sel = len(x_idx)
        patch = np.full((bands, len(target_times), H_sel, W_sel), np.nan, dtype=np.float32)

        for j, s1_idx in enumerate(matched_indices):
            if s1_idx is not None:
                tile = s1_array[:, s1_idx, :, :]
                tile = np.take(tile, y_idx, axis=1)
                tile = np.take(tile, x_idx, axis=2)
                patch[:, j] = tile

        # --- Valid mask before fill ---
        valid_mask = ~np.isnan(patch)


        # --- Fill NaNs with per-sample, per-band mean ---
        means = np.nanmean(patch, axis=(1, 2, 3), keepdims=True)
        means = np.where(np.isnan(means), 0.0, means)
        patch = np.where(valid_mask, patch, means)

        patches.append(patch)
        coords_time.append(np.array(matched_times))
        coords_x.append(s2_coords["x"][i])
        coords_y.append(s2_coords["y"][i])
        valid_masks.append(valid_mask)

    s1_patches_np = np.stack(patches, axis=0)           # (N, bands, T_sel, H, W)
    s1_valid_mask_np = np.stack(valid_masks, axis=0)    # same shape

    print(f's1 valid mask: {s1_valid_mask_np.shape}')
    s1_coords = {
        "time": np.stack(coords_time, axis=0),          # (N, T_sel)
        "x": np.stack(coords_x, axis=0),                # (N, W)
        "y": np.stack(coords_y, axis=0)                 # (N, H)
    }

    return s1_patches_np, s1_coords, s1_valid_mask_np


def verify_patches_against_cube(
    da: xr.DataArray,
    patches: torch.Tensor,
    coords_out: Dict[str, np.ndarray],
    n_samples: int = 5
):
    """
    Verifies extracted patches against the original xarray cube.

    Args:
        da: xarray.DataArray with dims (band, time, y, x)
        patches: torch.Tensor of shape (N, bands, select_t, h, w)
        coords_out: dict with keys 'time', 'y', 'x'
        n_samples: number of random samples to verify
    """
    print(f"\n🔍 Verifying {n_samples} random patches...")

    N = patches.shape[0]
    sample_ids = np.random.choice(N, size=n_samples, replace=False)

    for idx in sample_ids:
        t_coords = coords_out["time"][idx]
        y_coords = coords_out["y"][idx]
        x_coords = coords_out["x"][idx]

        #y0, y1 = (y_min, y_max) if da.y[0] <= da.y[-1] else (y_max, y_min)
        #x0, x1 = (x_min, x_max) if da.x[0] <= da.x[-1] else (x_max, x_min)
        y0, y1 = y_coords[0], y_coords[-1]
        x0, x1 = x_coords[0], x_coords[-1]
        # Select original patch using coordinate values
        da_patch = da.sel(
            time=xr.DataArray(t_coords, dims="time"),
            y=slice(y0, y1),
            x=slice(x0, x1),
        ).transpose("time", "index", "y", "x")

        print(f'{da_patch.shape}')

        da_patch_np = da_patch.values
        #da_patch_np = da_patch_np[5, 0, :, :]
        extracted_np = patches[idx].numpy()
        #extracted_np = extracted_np[5, 0, :, :]

        vec1 = np.sort(da_patch_np.flatten())
        vec2 = np.sort(extracted_np.flatten())

        is_equal = np.allclose(vec1, vec2, equal_nan=True)

        #is_equal = np.allclose(da_patch_np, extracted_np, equal_nan=True)

        print(f"🧪 Patch {idx}: {'✅ MATCH' if is_equal else '❌ MISMATCH'}")

    print("🔁 Verification complete.\n")


def extract_sentinel_patches(
    s2_array: np.ndarray,
    time_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    logger_name: Optional[str] = None,
    time_coords_2: np.ndarray = None,
    time_win: int = 20,
    h_win: int = 15,
    w_win: int = 15,
    time_stride: int = 17,
    h_stride: int = 9,
    w_stride: int = 9,
    select_t: int = 11,
    layout: str = 'BTYX',  # TBYX
    max_total_gap: int = 200,
    inference: bool = False,
) -> Tuple[torch.Tensor, Dict[str, np.ndarray], torch.Tensor, bool]:
    """
    Extract spatiotemporal patches from Sentinel-1/2 array using torch,
    randomly select `select_t` of `time_win` timesteps per patch,
    and return a validity mask (True = valid data).

    Args:
        s2_array: np.ndarray (bands, time, height, width)
        time_coords: np.ndarray (T,)
        y_coords: np.ndarray (H,)
        x_coords: np.ndarray (W,)
        time_win: temporal patch size
        h_win: spatial height of patch
        w_win: spatial width of patch
        *_stride: stride for each dimension
        select_t: number of timestamps to keep (≤ time_win)

    Returns:
        patches: (N, select_t, bands, h_win, w_win) torch.Tensor
        coords: dict with keys 'time' (N, select_t), 'y' (N, h_win), 'x' (N, w_win)
        valid_mask: torch.BoolTensor, True where data is valid, shape same as patches
    """
    assert select_t <= time_win, "Cannot select more timestamps than available."

    log = logging.getLogger(logger_name) if logger_name else None

    def regularize_time_rows(time_array: np.ndarray, row_mask: np.ndarray) -> np.ndarray:
        if not row_mask.any():
            return time_array

        fixed = time_array.copy()
        center_idx = fixed.shape[1] // 2
        offsets = np.arange(fixed.shape[1], dtype=np.int64) - center_idx
        one_day_steps = offsets.astype("timedelta64[D]").astype("timedelta64[ns]")
        centers = fixed[row_mask, center_idx].astype("datetime64[ns]")
        fixed[row_mask] = centers[:, None] + one_day_steps[None, :]
        return fixed

    def emit(message: str, *args, level: int = logging.DEBUG) -> None:
        if log is not None:
            log.log(level, message, *args)
        else:
            print(message % args if args else message)

    time_start_patch = time.time()
    emit("Converting input array to torch tensor")
    tensor = torch.from_numpy(s2_array).unsqueeze(0)  # (1, bands, T, H, W)
    if layout == 'BTYX': bands, T, H, W = s2_array.shape
    else: T, bands, H, W = s2_array.shape
    if time_win > T: time_win = T

    rm_unvalid = False

    emit("Input shape: bands=%s time=%s height=%s width=%s", bands, T, H, W)
    Nt = (T - time_win) // time_stride + 1
    Ny = (H - h_win) // h_stride + 1
    Nx = (W - w_win) // w_stride + 1
    emit("Extracting patches: Nt=%s Ny=%s Nx=%s", Nt, Ny, Nx)
    # Extract all patches using unfold
    if layout == 'BTYX': patches = tensor.unfold(2, time_win, time_stride)
    else: patches = tensor.unfold(1, time_win, time_stride)
    patches = patches.unfold(3, h_win, h_stride) \
        .unfold(4, w_win, w_stride)  # (1, bands, Nt, Ny, Nx, time_win, h_win, w_win) # (1, Nt, bands, Ny, Nx, time_win, h_win, w_win)

    if layout == 'BTYX': patches = patches.squeeze(0).permute(1, 2, 3, 0, 4, 5, 6)  # (bands, Nt, Ny, Nx, time, h, w)
    else: patches = patches.squeeze(0).permute(0, 2, 3, 1, 4, 5, 6)  # (Nt, Ny, Nx, bands, time, h, w)

    patches = patches.reshape(-1, bands, time_win, h_win, w_win)  # (N, bands, time, h, w)

    N = patches.shape[0]
    emit("Total patches extracted: %s (%.3fs)", N, time.time() - time_start_patch)
    # Randomly select select_t of time_win timesteps per patch
    time_rand_sel = time.time()
    emit("Selecting %s random timesteps from each patch", select_t)
    if select_t == time_win:
        random_idx = None
        selected_patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    else:
        random_idx = np.array([
            np.sort(np.random.choice(time_win, select_t, replace=False))
            for _ in range(N)
        ])
        random_idx_torch = torch.from_numpy(random_idx).long()

        # Select corresponding temporal slices using advanced indexing
        idx_batch = torch.arange(N).unsqueeze(1)  # (N, 1)
        selected_patches = patches[idx_batch, :, random_idx_torch]


    # === Compute valid mask: True if NOT NaN ===
    emit(
        "Final patch shape %s elapsed_seconds=%.3f",
        selected_patches.shape,
        time.time() - time_rand_sel,
    )
    time_val = time.time()
    valid_mask = ~torch.isnan(selected_patches)  # shape: (N, bands, select_t, h, w)
    emit("Validity mask computed (%.3fs)", time.time() - time_val)

    # === Filter out low-quality patches BEFORE filling ===
    emit("Filtering out low-quality patches")
    time_low = time.time()

    h_center = valid_mask.shape[3] // 2
    w_center = valid_mask.shape[4] // 2
    t_center = valid_mask.shape[1] // 2

    valid_patch_mask = valid_mask[:, t_center, :10, h_center, w_center].any(dim=1)  # (N, bands)

    # -- apply mask BEFORE filling NaNs --
    selected_patches = selected_patches[valid_patch_mask]
    valid_mask = valid_mask[valid_patch_mask]
    if random_idx is not None:
        random_idx = random_idx[valid_patch_mask.cpu().numpy()]
    emit(
        "Removed %s invalid patches in %.3fs",
        (~valid_patch_mask).sum().item(),
        time.time() - time_low,
    )
    emit("Remaining patches: %s", selected_patches.shape[0])

    # === Fill NaNs only if needed ===
    time_fill = time.time()
    if not valid_mask.all():
        emit("NaNs detected, filling missing values with sample mean per patch and band")
        sum_valid = torch.nan_to_num(selected_patches, nan=0.0, posinf=0.0, neginf=0.0).sum(dim=(1, 3, 4), keepdim=True)
        count_valid = valid_mask.sum(dim=(1, 3, 4), keepdim=True).clamp(min=1)
        mean_per_patch_band = sum_valid / count_valid  # (N, 1, bands, 1, 1)
        # Fill NaNs in-place without cloning
        selected_patches = torch.where(valid_mask, selected_patches, mean_per_patch_band)
        emit("NaNs filled with patch-band means (%.3fs)", time.time() - time_fill)
    else:
        emit("No NaNs found, skipping filling (%.3fs)", time.time() - time_fill)

    # === Compute corresponding coordinates ===
    emit("Computing coordinate ranges for all patches")
    time_range = time.time()
    # Get full 3D index grid (Nt, Ny, Nx)
    t_idx, y_idx, x_idx = np.meshgrid(
        np.arange(Nt), np.arange(Ny), np.arange(Nx), indexing='ij'
    )

    # Flatten in the same order as patches were reshaped
    t0_all = (t_idx * time_stride).reshape(-1)[valid_patch_mask.cpu().numpy()]  # (N_valid,)
    y0_all = (y_idx * h_stride).reshape(-1)[valid_patch_mask.cpu().numpy()]
    x0_all = (x_idx * w_stride).reshape(-1)[valid_patch_mask.cpu().numpy()]

    # Now guaranteed to match reshape(patches, bands, -1, ...)
    time_ranges = np.stack([time_coords[t0: t0 + time_win] for t0 in t0_all])
    if select_t == time_win:
        selected_time_coords = time_ranges
    else:
        selected_time_coords = np.take_along_axis(time_ranges, random_idx, axis=1)

    if time_coords_2 is not None:
        assert len(time_coords_2) == len(time_coords), \
            "time_coords_2 must have the same length as time_coords."
        time_ranges_add = np.stack([time_coords_2[t0: t0 + time_win] for t0 in t0_all])  # (N_valid, time_win)
        if select_t == time_win:
            selected_time_coords_add = time_ranges_add
        else:
            selected_time_coords_add = np.take_along_axis(time_ranges_add, random_idx, axis=1)  # (N_valid, select_t)
    else:
        selected_time_coords_add = None


    y_ranges = np.stack([y_coords[y0: y0 + h_win] for y0 in y0_all])
    x_ranges = np.stack([x_coords[x0: x0 + w_win] for x0 in x0_all])
    coords = {
        "time": selected_time_coords,  # (N, select_t)
        "y": y_ranges,                 # (N, h_win)
        "x": x_ranges                 # (N, w_win)
    }
    if selected_time_coords_add is not None:
        coords["time_add"] = selected_time_coords_add  # (N_valid, select_t)
    emit("Coordinate ranges computed after %.3fs", time.time() - time_range)

    time_gaps_start = time.time()
    time_gaps = compute_time_gaps(selected_time_coords)  # (N, 10)
    gap_mask = (time_gaps.sum(dim=1) < max_total_gap)  # (N,)

    if inference:
        bad_mask = ~gap_mask
        if bad_mask.any():
            bad_mask_np = bad_mask.cpu().numpy()
            selected_time_coords = regularize_time_rows(selected_time_coords, bad_mask_np)
            coords["time"] = selected_time_coords
            if "time_add" in coords:
                coords["time_add"] = regularize_time_rows(coords["time_add"], bad_mask_np)
    else:
        removed = (~gap_mask).sum().item()
        if removed:
            emit("Removing %s samples with total gaps > %s", removed, max_total_gap)
            rm_unvalid = True

        # apply mask to tensors
        selected_patches = selected_patches[gap_mask]
        valid_mask = valid_mask[gap_mask]

        # apply mask to numpy arrays
        idx_np = gap_mask.cpu().numpy()
        coords = {
            "time": coords["time"][idx_np],
            "y": coords["y"][idx_np],
            "x": coords["x"][idx_np],
            **({"time_add": coords["time_add"][idx_np]} if "time_add" in coords else {}),
        }

    emit("Time gaps computed after %.3fs", time.time() - time_gaps_start)
    emit("Extraction complete")
    return selected_patches, coords, valid_mask, rm_unvalid






def merge_s1_s2(
        cube_num: str,
        base_path: str = '/net/data_ssd/deepfeatures/trainingcubes',
        chunks: dict = None,
        save_path: str | None = None,
        var_name: str = "bands",
        valid_fraction_threshold: float = 0.015,   # A: ≥ 1.5%
        center_test_size: int = 6,               # B: center 6x6
        min_center_valid_pixels: int = 1         # B: ≥ 1 valid px in 6x6
):

    if save_path and os.path.isdir(save_path):
        return xr.open_zarr(save_path, consolidated=True).bands

    cube_path = os.path.join(base_path, cube_num + '.zarr')
    ds = xr.open_zarr(cube_path)

    # Cloud-masked S2 (dims typically: band,time,y,x)
    da = ds.s2l2a.where(ds.cloud_mask == 0)

    # ---------- Criterion A: ≥ valid_fraction_threshold across (band,y,x) ----------
    n_total = da.sizes["band"] * da.sizes["y"] * da.sizes["x"]
    valid_data_count = da.notnull().sum(dim=("band", "y", "x"))
    frac_valid = valid_data_count / n_total
    critA = frac_valid > 0. # = valid_fraction_threshold  # (time,)


    # ---------- Keep times that satisfy A ----------
    keep_mask = critA
    keep_mask_np = keep_mask.compute().values.astype(bool)  # -> NumPy 1D

    keep_idx = np.nonzero(keep_mask_np)[0]
    da = da.isel(time=keep_idx)
    #da = da.sel(time=da.time.where(keep_mask.compute(), drop=True))

    # Chunking
    if chunks is None:
        chunks = {"time": 1, "y": da.sizes["y"], "x": da.sizes["x"]}
    da = da.chunk(chunks)

    # Prepare spectral data (your existing helper)
    da = prepare_spectral_data(da, to_ds=False)
    if "index" in da.dims:
        da = da.rename({"index": "band"})

    coords = {dim: da.coords[dim].values for dim in ("time", "y", "x") if dim in da.dims}

    # ----- Match S1 to S2 grid/times -----
    s1 = match_sentinel1_to_s2_cube(ds)  # expected: s1.backscatter(time, band, y, x)
    s1_times = s1.time.values
    s1_y = s1.y.values
    s1_x = s1.x.values

    s2_times_used = xr.DataArray(coords['time'], dims=["time"])
    s2_y_used     = xr.DataArray(coords['y'], dims=["y"])
    s2_x_used     = xr.DataArray(coords['x'], dims=["x"])

    nearest_t_idx = nearest_indices(s2_times_used.values, s1_times)
    nearest_y_idx = nearest_indices(s2_y_used.values,   s1_y)
    nearest_x_idx = nearest_indices(s2_x_used.values,   s1_x)

    s1_matched = s1.backscatter.isel(
        time=xr.DataArray(nearest_t_idx, dims=["time"]),
        y=xr.DataArray(nearest_y_idx, dims=["y"]),
        x=xr.DataArray(nearest_x_idx, dims=["x"])
    ).assign_coords(time=s2_times_used, y=s2_y_used, x=s2_x_used)

    # Keep only minimal coords/attrs and reorder to (band,time,y,x)
    allowed = {"band", "time", "y", "x"}
    extra_non_dim = [c for c in s1_matched.coords if c not in allowed and c not in s1_matched.dims]
    s1_matched = s1_matched.drop_vars(extra_non_dim).assign_attrs({})
    s1_BTYX = s1_matched.transpose("band", "time", "y", "x")

    # S2 is already band,time,y,x after prepare
    s2_BTYX = da

    # Concat along band
    combined_BTYX = xr.concat([s2_BTYX, s1_BTYX],
                              dim="band", coords="minimal", compat="override", join="outer")

    # Add s1_time coord aligned to S2 time (ns dtype)
    matched_s1_times = s1_times[nearest_t_idx].astype("datetime64[ns]")
    combined_BTYX = combined_BTYX.assign_coords(s1_time=("time", matched_s1_times))

    combined_BTYX = combined_BTYX.rename(var_name)

    if save_path:
        combined_BTYX.to_dataset(name=var_name).to_zarr(save_path, mode="w", consolidated=True)


        # Proactively free remote readers before returning
        for name in ['ds','da','s1','s1_matched','s2_BTYX','s1_BTYX','combined_BTYX']:
            if name in locals():
                del locals()[name]
        gc.collect()

        # Re-open a fresh handle pointing ONLY to the written Zarr
        reopened = xr.open_zarr(save_path, consolidated=True)[var_name]

        # Optional: rechunk to what you want downstream
        # reopened = reopened.chunk({"time": 1, "y": reopened.sizes["y"], "x": reopened.sizes["x"]})

        return reopened

    return combined_BTYX
