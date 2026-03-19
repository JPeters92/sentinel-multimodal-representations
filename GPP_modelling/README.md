# GPP Modelling

This directory contains the GPP regression workflow built on top of daily feature cubes and FLUXNET targets.

### Prerequisites:

- the base feature cubes `s1_s2_<cid>.zarr` must already be created with [`feature_cube/feature_cube_torch.py`](../feature_cube/feature_cube_torch.py)
- the Warm Winter 2020 ecosystem dataset must be available:
  `https://www.icos-cp.eu/data-products/2G60-ZHAK`

### Cube Mapping

The cube IDs used in this module correspond to the Sentinel cubes and are mapped to the corresponding FLUXNET site codes in [`sites.py`](sites.py); this mapping is used throughout the workflow.
If the cubes are re-sampled or a new sampling is created, this mapping must be updated accordingly.


### 1. Cube and flux exploration

Run [`validate_feature_cubes.py`](validate_feature_cubes.py).

Purpose:
- finds cubes that have both feature data and flux coverage
- reports missing feature cubes or missing flux sites
- reports valid timestamps per cube for the relevant years

Use this first when deciding which `CUBE_IDS` to include in dataset generation.

### 2. Feature interpolation

Choose one feature source:

- [`linear.py`](linear.py)
  Reads `s1_s2_<cid>.zarr`, computes spatial means, applies linear interpolation, and writes `s1_s2_<cid>_mean_linear.zarr`.

- [`kalman.py`](kalman.py)
  Reads `s1_s2_<cid>.zarr`, computes spatial means, applies UCM-based gap filling, appends daily potential radiation, and writes `s1_s2_<cid>_mean_ucm_flux.zarr`.

The prepared feature source is then consumed by [`dataset.py`](dataset.py).

### 3. Training set creation

Run [`dataset.py`](dataset.py).

Purpose:
- loads the prepared feature cubes and the daily GPP from FLUXNET WarmWinter files
- applies the temporal split  of 90 days into train and validation years
- writes model input tensors `.npz` and sample metadata for the generated windows `.csv`

Outputs:
- `gpp_90day_samples_..._train.npz`
- `gpp_90day_samples_..._val.npz`

### 4. Model training

Run [`GPP_train.py`](GPP_train.py).

Purpose:
- loads the generated dataset from [`dataset.py`](dataset.py) via [`GPP_loader.py`](GPP_loader.py)
- trains the Transformer regressor from [`model.py`](model.py)
- runs the configured hyperparameter grid search


### 5. Evaluation and visualization

Run [`GPP_plot.py`](GPP_plot.py).

Purpose:
- predicts GPP for selected cubes using the trained model
- creates per-cube and combined multi-site plots
- reports RMSE and NRMSE

This is the main evaluation and visualization script in the directory.

## Visualization

The current combined comparison figure is:

- [`gpp_compare_combined_7feat_bottomlegend.png`](gpp_compare_out/gpp_compare_combined_7feat_bottomlegend.png)

<img src="./gpp_compare_out/gpp_compare_combined_7feat_bottomlegend.png" alt="Combined 7-feature GPP comparison" width="270">


This is the visualization produced by [`GPP_plot.py`](GPP_plot.py).
