# Feature Cube Module

This module contains the scripts for generating and inspecting the latent feature cubes used in the downstream workflows.

## Scope

The main output of this module is a feature cube stored as a `.zarr` dataset. These cubes are created from the trained representation models and are later used, for example, in the GPP use case.

## Scripts

### 1. Feature cube extraction

Execute [`feature_cube_torch.py`](feature_cube_torch.py). The script supports argparse-based runtime configuration.

Default CLI settings:
- `--cuda-device cuda:0`                                            # GPU used for inference
- `--batch-size 64`                                                 # Number of patches per forward pass
- `--base-path /net/data/deepfeatures/trainingcubes`                # Input directory of the Sentinel-2 base cubes
- `--merged-path /net/data_ssd/deepfeatures/s1_s2_cubes`            # Cache directory for merged Sentinel-1/2 cubes
- `--output-path /net/data_ssd/deepfeatures/sciencecubes_processed` # Output directory for the final feature cubes
- `--checkpoint-path ../checkpoints/fusion/s1_s2/fuse_model.ckpt`   # Fusion checkpoint used for feature extraction
- `--processes 6`                                                   # Worker processes for patch preprocessing
- `--split-count 1`                                                 # Total number of processing splits
- `--split-index 0`                                                 # Index of the current split to execute
- `--space-block-size 90`                                           # Spatial block size per chunk
- `--log-level INFO`                                                # Logging level

Example CLI override:

```bash
python feature_cube_torch.py \
  --cuda-device cuda:1 \
  --batch-size 96 \
  --cube-ids 000003 000008 000022 \
  --processes 6 \
  --split-count 2 \
  --split-index 0 \
  --space-block-size 90 \
  --log-level DEBUG
```

This is the main feature extraction script for the fused Sentinel-1/2 setup. It
- merges Sentinel-1 and Sentinel-2 data on the shared grid,
- loads the trained fusion model,
- extracts spatio-temporal patches,
- computes latent features for the selected cube IDs,
- and writes the resulting feature cubes to `.zarr`.



### 2. Cube verification

Execute [`verify_cube.py`](verify_cube.py) for a quick inspection of an existing feature cube.

This script opens a `.zarr` feature cube and reports NaN statistics per timestep and over the full cube.

### 3. Paper visualization

Execute [`paper_visualisation.py`](paper_visualisation.py).

This script creates qualitative figures for selected cubes by comparing Sentinel-2 RGB views with PCA-based visualizations of the learned feature cube representations.

<img src="paper_vis/S1_S2_PCA.png" alt="Sentinel-2 and PCA-based feature cube visualization" width="600">
