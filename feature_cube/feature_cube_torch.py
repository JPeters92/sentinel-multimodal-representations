import argparse
import logging
import os
import pathlib
import sys
import time
import warnings
from multiprocessing import Lock, Pool, Value

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.preprocess_sentinel import extract_sentinel_patches, merge_s1_s2
from dataset.utils import compute_time_gaps, extract_center_coordinates
from model.model_fusion import FusedS1S2
from model.model_s1_s2 import TransformerAE


parser = argparse.ArgumentParser(
    description="Extract fused Sentinel-1/2 feature cubes from science or training cubes.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--cuda-device", default="cuda:0", help="Torch device used for inference.")
parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size for extracted patches.")
parser.add_argument(
    "--base-path",
    default="/net/data/deepfeatures/trainingcubes",
    help="Directory containing the source Sentinel-2 cubes used as the basis for S1/S2 merging.",
)
parser.add_argument(
    "--merged-path",
    default="/net/data_ssd/deepfeatures/s1_s2_cubes",
    help="Directory for cached intermediate S1/S2 merged cubes.",
)
parser.add_argument(
    "--output-path",
    default="/net/data_ssd/deepfeatures/sciencecubes_processed",
    help="Directory where the final latent feature cubes are written as .zarr stores.",
)
parser.add_argument(
    "--checkpoint-path",
    default="../checkpoints/fusion/fuse_model.ckpt",
    help="Fusion checkpoint used to generate the latent feature cubes.",
)
parser.add_argument("--processes", type=int, default=6, help="Number of worker processes for patch preprocessing.")
parser.add_argument("--split-count", type=int, default=1, help="Total number of spatial splits for distributed processing.")
parser.add_argument("--split-index", type=int, default=0, help="Index of the current split to process.")
parser.add_argument("--space-block-size", type=int, default=90, help="Spatial block size used during cube traversal.")
parser.add_argument(
    "--cube-ids",
    nargs="*",
    default=None,
    help="Optional list of cube IDs to process. If omitted, the internal default cube list is used.",
)
parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
args = parser.parse_args()

CUDA_DEVICE = args.cuda_device
BATCH_SIZE = args.batch_size
BASE_PATH = args.base_path
MERGED_PATH = args.merged_path
OUTPUT_PATH = args.output_path
CHECKPOINT_PATH = args.checkpoint_path
PROCESSES = args.processes
SPLIT_COUNT = args.split_count
SPLIT_INDEX = args.split_index
SPACE_BLOCK_SIZE = args.space_block_size

if SPLIT_COUNT < 1:
    raise ValueError("split-count must be at least 1")
if not 0 <= SPLIT_INDEX < SPLIT_COUNT:
    raise ValueError("split-index must be in the range [0, split-count)")

_log_level_str = str(args.log_level).upper()
if _log_level_str in ("DEBUG", "10"):
    LOG_LEVEL_INT = logging.DEBUG
else:
    LOG_LEVEL_INT = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL_INT,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL_INT)

_worker_id = None


def _init_worker(counter, lock, log_level_int):
    global _worker_id
    with lock:
        _worker_id = counter.value
        counter.value += 1

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(log_level_int)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level_int)
    logging.getLogger(logger.name).setLevel(log_level_int)


def _extract_worker(args_tuple):
    (
        data_sub,
        time_sub,
        y_sub,
        x_sub,
        time_sub_2,
        extractor_kwargs,
    ) = args_tuple

    try:
        interior = data_sub[:10, 5, 7:-7, 7:-7]
    except IndexError:
        return None

    if not np.any(~np.isnan(interior)):
        return None

    worker_logger_name = f"{logger.name}.W{_worker_id}" if _worker_id is not None else logger.name
    worker_logger = logging.getLogger(worker_logger_name)
    worker_logger.setLevel(LOG_LEVEL_INT)

    patches, coords, valid_mask, not_val = extract_sentinel_patches(
        data_sub,
        time_sub,
        y_sub,
        x_sub,
        logger_name=worker_logger_name,
        time_coords_2=time_sub_2,
        inference=True,
        **extractor_kwargs,
    )

    if patches is None or getattr(patches, "shape", (0,))[0] == 0:
        return None

    return patches, dict(coords), valid_mask, not_val


def extract_sentinel_patches_pool(
    data,
    time_coords,
    y_coords,
    x_coords,
    *,
    time_coords_2=None,
    processes=4,
    overlap=14,
    **kwargs,
):
    y_full = data.shape[2]
    if y_full == 0:
        return torch.empty((0,)), {}, torch.empty((0,)), True

    edges = np.linspace(0, y_full, processes + 1, dtype=int)
    extractor_kwargs = dict(kwargs)

    jobs = []
    for i in range(processes):
        y0 = int(edges[i])
        y1 = min(y_full, int(edges[i + 1]) + overlap)
        if y1 <= y0:
            continue

        jobs.append(
            (
                data[:, :, y0:y1, :],
                time_coords,
                y_coords[y0:y1],
                x_coords,
                time_coords_2,
                extractor_kwargs,
            )
        )

    if not jobs:
        return torch.empty((0,)), {}, torch.empty((0,)), True

    counter = Value("i", 0)
    lock = Lock()
    with Pool(processes=processes, initializer=_init_worker, initargs=(counter, lock, LOG_LEVEL_INT)) as pool:
        results = pool.map(_extract_worker, jobs)

    results = [result for result in results if result is not None]
    if not results:
        return torch.empty((0,)), {}, torch.empty((0,)), True

    patches_list, coords_list, masks_list, not_vals = zip(*results)
    patches_all = torch.cat(patches_list, dim=0)
    valid_mask_all = torch.cat(masks_list, dim=0)
    coords_all = {
        "time": np.concatenate([coords["time"] for coords in coords_list], axis=0),
        "y": np.concatenate([coords["y"] for coords in coords_list], axis=0),
        "x": np.concatenate([coords["x"] for coords in coords_list], axis=0),
    }
    if "time_add" in coords_list[0]:
        coords_all["time_add"] = np.concatenate([coords["time_add"] for coords in coords_list], axis=0)

    return patches_all, coords_all, valid_mask_all, all(not_vals)


def default_cube_ids():
    existing = {"043", "027", "442", "053", "034", "022", "368", "348", "356", "449", "212", "228"}
    all_nums = [f"{i:03d}" for i in range(500)]
    cube_nums = [num for num in all_nums if num not in existing and int(num) > 248]
    return [f"{int(num):06d}" for num in cube_nums]


def create_empty_dataset(feature_names, xs, ys, out_path, times=None, dtype=np.float32):
    if os.path.exists(out_path):
        return xr.open_zarr(out_path)

    times = np.asarray(times).astype("datetime64[ns]")
    data = np.full((len(feature_names), len(times), len(ys), len(xs)), np.nan, dtype=dtype)

    da = xr.DataArray(
        data,
        dims=("feature", "time", "y", "x"),
        coords={
            "feature": np.asarray(feature_names, dtype=str),
            "time": times,
            "y": np.asarray(ys),
            "x": np.asarray(xs),
        },
        name="features",
    )
    encoding = {"features": {"chunks": (len(feature_names), 1, len(ys), len(xs))}}
    ds_out = xr.Dataset({"features": da}).drop_vars("feature")
    ds_out.to_zarr(out_path, mode="w", encoding=encoding)
    return ds_out


def init_output_from_source(da, feature_names, out_path):
    if os.path.exists(out_path):
        ds0 = xr.open_zarr(out_path, consolidated=True)
        feats = ds0["features"]
        reduce_dims = tuple(dim for dim in feats.dims if dim != "time")
        empty_mask_np = feats.isnull().all(dim=reduce_dims).compute().values.astype(bool)
        empty_times = ds0["time"].values[empty_mask_np]
        xs = ds0["x"].values
        ys = ds0["y"].values
        t_ns = ds0["time"].values.astype("datetime64[ns]")
        time_to_idx = {int(t.astype("int64")): i for i, t in enumerate(t_ns)}
        return ds0, empty_times, xs, ys, time_to_idx

    da_c = da.isel(time=slice(5, -5), y=slice(7, -7), x=slice(7, -7), band=slice(0, 10))
    complete_px = np.isfinite(da_c).all(dim="band")
    frac_complete = complete_px.mean(dim=("y", "x"))
    keep_mask = (frac_complete >= 0.035).compute().values
    ok_idx = np.flatnonzero(keep_mask)
    times_ok_ns = np.asarray(da_c.time.values)[ok_idx]

    global_xs = da.x.values[7:-7]
    global_ys = da.y.values[7:-7]
    ds0 = create_empty_dataset(
        feature_names=feature_names,
        xs=global_xs,
        ys=global_ys,
        out_path=out_path,
        times=times_ok_ns,
        dtype=np.float32,
    )
    t_ns = ds0["time"].values.astype("datetime64[ns]")
    time_to_idx = {int(t.astype("int64")): i for i, t in enumerate(t_ns)}
    return ds0, times_ok_ns, global_xs, global_ys, time_to_idx


def reset_frame():
    global current_canvas, filled_once, current_time
    current_canvas[:] = np.nan
    filled_once[:] = False
    current_time = None


def flush_frame(canvas, f_ds, out_path, time_value, time_to_idx):
    if time_value is None:
        return False

    t_ns = np.datetime64(time_value, "ns")
    key = int(t_ns.astype("int64"))
    if key not in time_to_idx:
        logger.info("FLUSH: skip %s because it is not part of the target timeline", t_ns)
        return False

    idx = time_to_idx[key]
    da = xr.DataArray(
        canvas[:, np.newaxis, :, :],
        dims=("feature", "time", "y", "x"),
        coords={
            "feature": f_ds.feature.values,
            "time": f_ds.time.values[idx:idx + 1],
            "y": f_ds.y.values,
            "x": f_ds.x.values,
        },
        name="features",
    )
    ds = xr.Dataset({"features": da}).drop_vars("feature")
    ds.to_zarr(
        out_path,
        mode="r+",
        region={
            "time": slice(idx, idx + 1),
            "y": slice(0, len(f_ds.y)),
            "x": slice(0, len(f_ds.x)),
        },
    )
    return True


def coord_to_idx(vals, mapping, axis_vals):
    vals = np.asarray(vals)
    idxs = np.empty(vals.shape, dtype=np.int64)
    for j, value in enumerate(vals):
        fval = float(value)
        if fval in mapping:
            idxs[j] = mapping[fval]
        else:
            idxs[j] = int(np.argmin(np.abs(axis_vals - value)))
    return idxs


class XrFeatureDataset:
    def __init__(
        self,
        data_cube: xr.DataArray,
        matched_s1_times,
        times_ok_ns,
        time_block_size=11,
        space_block_size=SPACE_BLOCK_SIZE,
        time_overlap=10,
        space_overlap=14,
        split_count=1,
        split_index=0,
    ):
        self.data_cube = data_cube
        self.matched_s1_times = matched_s1_times
        self.time_block_size = time_block_size
        self.space_block_size = space_block_size
        self.time_overlap = time_overlap
        self.space_overlap = space_overlap
        self.split_count = max(1, int(split_count))
        self.split_index = int(split_index)

        self.time_len = int(data_cube.sizes["time"])
        self.y_len = int(data_cube.sizes["y"])
        self.x_len = int(data_cube.sizes["x"])
        logger.info(
            "ScienceCube bounds: y_len=%s x_len=%s time_len=%s",
            self.y_len,
            self.x_len,
            self.time_len,
        )

        self.save_frame = True
        self.chunks_bounds = self.compute_bounds(time_slide=True, time_block=self.time_block_size, space_block=self.space_block_size)
        self.chunk_split = self._compute_chunk_split()
        self.chunk_idx, self.max_chunk = self._compute_split_chunk_range(
            total_chunks=len(self.chunks_bounds),
            split_count=self.split_count,
            split_index=self.split_index,
        )
        logger.info(
            "Chunks to process: %s / %s (range %s..%s)",
            max(0, self.max_chunk - self.chunk_idx),
            len(self.chunks_bounds),
            self.chunk_idx,
            self.max_chunk,
        )

        self.times_ok_ns = np.asarray(times_ok_ns).astype("datetime64[ns]")
        self._times_ok_set = set(self.times_ok_ns.astype("int64").tolist())

    def __iter__(self):
        return self

    def _nominal_ranges(self, n, block, sliding=False):
        if sliding:
            return [(i, i + block) for i in range(0, n - block + 1, 1)]
        return [(i, i + block) for i in range(0, n, block) if i + block <= n]

    def _compute_chunk_split(self):
        y_nom = self._nominal_ranges(self.y_len, self.space_block_size)
        x_nom = self._nominal_ranges(self.x_len, self.space_block_size)
        return max(1, len(y_nom) * len(x_nom))

    def compute_bounds(self, time_slide, time_block, space_block, split_chunk=False):
        if split_chunk and self.chunk_idx == 0:
            t_len = self.time_block_size
        elif split_chunk and self.chunk_idx > 0:
            t_len = self.time_block_size + 10
        else:
            t_len = self.time_len

        t_nom = self._nominal_ranges(t_len, time_block, sliding=time_slide)
        y_nom = self._nominal_ranges(self.y_len, space_block)
        x_nom = self._nominal_ranges(self.x_len, space_block)

        chunks = []
        for (t0_nom, t1_nom) in t_nom:
            t0 = t0_nom if time_slide else (t0_nom - self.time_overlap if t0_nom > 0 else t0_nom)
            t1 = t1_nom
            t0 = max(0, t0)
            t1 = min(t_len, t1)

            for (y0_nom, y1_nom) in y_nom:
                y0 = y0_nom - self.space_overlap if y0_nom > 0 else y0_nom
                y1 = y1_nom
                y0 = max(0, y0)
                y1 = min(self.y_len, y1)

                for (x0_nom, x1_nom) in x_nom:
                    x0 = x0_nom - self.space_overlap if x0_nom > 0 else x0_nom
                    x1 = x1_nom
                    x0 = max(0, x0)
                    x1 = min(self.x_len, x1)
                    chunks.append((t0, t1, y0, y1, x0, x1))

        return chunks

    def _compute_split_chunk_range(self, total_chunks, split_count, split_index):
        if total_chunks <= 0:
            return 0, 0
        frames_total = total_chunks // self.chunk_split
        frames_per_split = frames_total // split_count
        start_frame = split_index * frames_per_split
        end_frame = (split_index + 1) * frames_per_split
        return start_frame * self.chunk_split, min(end_frame * self.chunk_split, total_chunks)

    def __next__(self):
        warnings.filterwarnings("ignore")
        if self.chunk_idx >= self.max_chunk:
            raise StopIteration

        t0, t1, y0, y1, x0, x1 = self.chunks_bounds[self.chunk_idx]
        logger.info("Getting chunk: time=%s-%s y=%s-%s x=%s-%s", t0, t1, y0, y1, x0, x1)

        chunk = self.data_cube.isel(time=slice(t0, t1), y=slice(y0, y1), x=slice(x0, x1))
        coords = {key: chunk.coords[key].values for key in chunk.coords}

        center_time = np.datetime64(coords["time"][coords["time"].size // 2]).astype("datetime64[ns]")
        if int(center_time.astype("int64")) not in self._times_ok_set:
            logger.info("Skipping chunk %s because center time %s is not in the target timeline", self.chunk_idx, center_time)
            self.chunk_idx = ((self.chunk_idx // self.chunk_split) + 1) * self.chunk_split - 1
            self.save_frame = False
            return None, None, None, None, None, None

        start_values = time.time()
        data = chunk.values
        logger.info("Chunk values computed in %.3fs", time.time() - start_values)

        valid_pixel_mask = np.isnan(data[:10, 5, 7:-7, 7:-7])
        non_nan_count = (~valid_pixel_mask).sum()
        nan_count = valid_pixel_mask.sum()
        logger.info("Chunk NaNs: %s Non-NaNs: %s", f"{nan_count:,}", f"{non_nan_count:,}")

        if non_nan_count == 0:
            return None, None, None, None, None, None

        start_split = time.time()
        patches_all, coords_all, valid_mask_all, not_val = extract_sentinel_patches_pool(
            data,
            coords["time"],
            coords["y"],
            coords["x"],
            time_coords_2=self.matched_s1_times[t0:t1],
            processes=PROCESSES,
            time_win=11,
            time_stride=1,
            h_stride=1,
            w_stride=1,
        )
        logger.info("Patches preprocessed in %.3fs", time.time() - start_split)

        if patches_all.shape[0] == 0 and not_val:
            self.save_frame = False
        if patches_all.shape[0] == 0:
            return None, None, None, None, None, None

        time_gaps_s2 = compute_time_gaps(coords_all["time"])
        time_gaps_s1 = compute_time_gaps(coords_all["time_add"])
        time_add = coords_all["time_add"][:, 5].astype("datetime64[D]").astype("int64")
        time_ref = coords_all["time"][:, 5].astype("datetime64[D]").astype("int64")
        time_gaps_c = torch.abs(torch.from_numpy(time_add - time_ref)).view(-1, 1)
        return patches_all, coords_all, valid_mask_all, time_gaps_s1, time_gaps_s2, time_gaps_c


def build_model(device):
    ae_s1 = TransformerAE(dbottleneck=2, channels=2, num_reduced_tokens=7).eval()
    ae_s2 = TransformerAE(dbottleneck=9, channels=10, num_reduced_tokens=6).eval()
    model = FusedS1S2(
        enc_s1=ae_s1.encoder,
        dec_s1=ae_s1.decoder,
        enc_s2=ae_s2.encoder,
        dec_s2=ae_s2.decoder,
        dbottleneck_s1=2,
        dbottleneck_s2=9,
        freeze_encoders=False,
        dbottleneck=7,
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def process_cube(cube_num, model, device):
    global current_canvas, filled_once, current_time

    global_mae_sum = 0.0
    global_mape_sum = 0.0
    global_count = 0
    eps = 1e-6

    logger.info("Processing cube %s", cube_num)
    merged_path = os.path.join(MERGED_PATH, f"s1_s2_{cube_num}.zarr")
    ds = merge_s1_s2(cube_num, base_path=BASE_PATH, save_path=merged_path)

    matched_s1_times = ds["s1_time"].values
    feature_names = ["F01", "F02", "F03", "F04", "F05", "F06", "F07"]
    output_path = os.path.join(OUTPUT_PATH, f"s1_s2_{cube_num}.zarr")

    ds0, times_ok_ns, global_xs, global_ys, time_to_idx = init_output_from_source(ds, feature_names, output_path)
    logger.info("times_ok_ns count: %s", len(times_ok_ns))

    x_to_idx = {float(value): i for i, value in enumerate(global_xs)}
    y_to_idx = {float(value): i for i, value in enumerate(global_ys)}

    current_canvas = np.full((len(feature_names), len(global_ys), len(global_xs)), np.nan, dtype=np.float32)
    filled_once = np.zeros((len(global_ys), len(global_xs)), dtype=bool)
    current_time = None

    dataset = XrFeatureDataset(
        data_cube=ds,
        matched_s1_times=matched_s1_times,
        times_ok_ns=times_ok_ns,
        split_count=SPLIT_COUNT,
        split_index=SPLIT_INDEX,
        space_block_size=SPACE_BLOCK_SIZE,
    )

    chunk_processed_time = time.time()
    for _, chunk in enumerate(dataset):
        mae_sum = 0.0
        mape_sum = 0.0
        count = 0
        start_time = time.time()
        logger.info("Chunk %s received in %.2fs", dataset.chunk_idx, start_time - chunk_processed_time)

        processed_data, coords, valid_mask, time_gaps_s1, time_gaps_s2, time_gaps_c = chunk
        if processed_data is None:
            n_samples = 0
        else:
            n_samples = processed_data.shape[0]
            center_time, center_xs, center_ys = extract_center_coordinates(coords)
            if current_time is None:
                current_time = center_time

        for start in tqdm(range(0, n_samples, BATCH_SIZE), desc="Reconstructing", unit="batch"):
            end = min(start + BATCH_SIZE, n_samples)
            batch_processed = processed_data[start:end].to(device, dtype=torch.float32)
            batch_mask = valid_mask[start:end].to(device, dtype=torch.bool)
            batch_s1 = time_gaps_s1[start:end].to(device, dtype=torch.int32)
            batch_s2 = time_gaps_s2[start:end].to(device, dtype=torch.int32)
            batch_c = time_gaps_c[start:end].to(device, dtype=torch.int32)

            x_s2 = batch_processed[:, :, :10, :, :]
            x_s1 = batch_processed[:, :, 10:, :, :]
            y_s1, y_s2, zf = model((x_s1, x_s2, batch_s1, batch_s2, batch_c))
            y_all = torch.cat([y_s2, y_s1], dim=2)

            bsz, t_len, _, height, width = batch_processed.shape
            ct, cx, cy = t_len // 2, height // 2, width // 2
            central_in = batch_processed[:, ct, :, cx, cy]
            central_out = y_all[:, ct, :, cx, cy]
            central_mask = batch_mask[:, ct, :, cx, cy]

            bx = center_xs[start:end]
            by = center_ys[start:end]
            x_idx = coord_to_idx(bx, x_to_idx, global_xs)
            y_idx = coord_to_idx(by, y_to_idx, global_ys)

            current_canvas[:, y_idx, x_idx] = zf.detach().cpu().numpy().astype(np.float32).T
            filled_once[y_idx, x_idx] = True

            valid_in = central_in[central_mask]
            valid_out = central_out[central_mask]
            diff = (valid_out - valid_in).abs()
            mae_sum += diff.sum().item()
            mape_sum += (diff / valid_in.abs().clamp_min(eps)).sum().item()
            count += central_mask.sum().item()

        chunk_processed_time = time.time()
        logger.info("Chunk %s, cube=%s processed in %.3fs", dataset.chunk_idx, cube_num, chunk_processed_time - start_time)

        chunk_mae = mae_sum / max(count, 1)
        chunk_mape = 100.0 * mape_sum / max(count, 1)
        global_mae_sum += mae_sum
        global_mape_sum += 100.0 * mape_sum
        global_count += count

        if (dataset.chunk_idx + 1) % dataset.chunk_split == 0:
            if dataset.save_frame:
                saved = flush_frame(current_canvas, ds0, output_path, current_time, time_to_idx)
                frame_label = np.datetime_as_string(current_time, unit="D") if current_time is not None else "None"
                logger.info(
                    "Frame: date=%s status=%s",
                    frame_label,
                    "saved" if saved else "skipped",
                )
                reset_frame()
            else:
                dataset.save_frame = True

        dataset.chunk_idx += 1
        logger.info("Central-pixel Chunk MAE: %.6f", chunk_mae)
        logger.info("Central-pixel Chunk MAPE: %.4f%%", chunk_mape)
        logger.info("Iteration ended in %.3fs", time.time() - start_time)

    logger.info("Central-pixel Global MAE: %.6f", global_mae_sum / max(global_count, 1))
    logger.info("Central-pixel Global MAPE: %.4f%%", global_mape_sum / max(global_count, 1))


def main():
    os.makedirs(MERGED_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    device = torch.device(CUDA_DEVICE)
    model = build_model(device)
    cube_nums = args.cube_ids if args.cube_ids else default_cube_ids()
    logger.info("Cube count: %s", len(cube_nums))

    for cube_num in cube_nums:
        process_cube(cube_num, model, device)


if __name__ == "__main__":
    main()
