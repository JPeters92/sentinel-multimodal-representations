# Context-Aware Multimodal Representation Learning for Spatio-Temporally Explicit Environmental Modelling

Spatio-temporally explicit embeddings are latent representations that preserve where and when environmental observations were made, 
enabling ecosystem dynamics to be modelled across fine spatial scales and varying temporal conditions. 
This repository implements a two-stage representation learning framework to create such embeddings from Earth observation data, 
introduced here for Sentinel-1 and Sentinel-2. It focuses on a context-aware multimodal setup that preserves high spatial and temporal fidelity while combining complementary sensor modalities. The framework first learns modality-specific representations and then combines them in a shared fusion model. 
The resulting latent feature cubes are designed as analysis-ready datasets for downstream environmental modelling tasks such as Gross Primary Production (GPP).

<p align="center">
  <img src="images/embeddings.png" alt="Sentinel-2 RGB and PCA feature pairs" width="44%">
  <img src="images/gpp_compare_combined_7feat_bottomlegend.png" alt="GPP prediction comparison" width="38%">
</p>
<p align="center">
  <sub>Left: Sentinel-2 RGB views paired with PCA projections of learned feature representations. Right: Observed daily GPP and predictions from seven learned features.</sub>
</p>

## Installation

We recommend using Python 3.12 and CUDA 12.1.

Clone the repository and create the conda environment:

```bash
git clone git@github.com:Julia310/sentinel-multimodal-representations.git
cd sentinel-spatiotemporal-representations
conda env create -f environment.yml
conda activate emb-venv
```
## Workflow

The repository is organized into five main modules:

- [`dataset`](dataset/README.md): data access, preprocessing, array preparation, and HDF5 dataset creation
- [`model`](model/README.md): modality-specific and fusion model definitions, losses, and architectural components
- [`training`](training/README.md): modality-specific training, fusion training, validation, and loss visualization
- [`feature_cube`](feature_cube/README.md): generation and inspection of latent feature cubes
- [`GPP_modelling`](GPP_modelling/README.md): downstream GPP use case built on the learned feature cubes

The core workflow is:

1. [`dataset/train_dataset.py`](dataset/train_dataset.py) to prepare aligned Sentinel-based HDF5 training data
2. [`training/train_modality.py`](training/train_modality.py) for `s1` and `s2` to train the modality-specific encoders and decoders
3. [`training/train_fusion.py`](training/train_fusion.py) to train the shared fusion model on aligned Sentinel-1 and Sentinel-2 inputs
4. [`feature_cube/feature_cube_torch.py`](feature_cube/feature_cube_torch.py) to generate latent feature cubes for downstream tasks

For downstream evaluation such as GPP modelling, continue in [`GPP_modelling`](GPP_modelling/README.md).

## Attribution

```bibtex
@article{peters2025context,
  title={Context-Aware Multimodal Representation Learning for Spatio-Temporally Explicit Environmental Modelling},
  author={Peters, Julia and Mora, Karin and Mahecha, Miguel D and Ji, Chaonan and Montero, David and Mosig, Clemens and Kraemer, Guido},
  journal={arXiv preprint arXiv:2511.11706},
  year={2025}
}
```
