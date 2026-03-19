# Model Module

This module contains the model definitions for feature extraction. It covers the modality-specific autoencoders, the fusion model, the shared building blocks, and the reconstruction losses.

### 1. Modality-specific autoencoder

Use [`model_s1_s2.py`](model_s1_s2.py).

Purpose:
- entry point for modality-specific pretraining
- defines the Transformer-based autoencoder used for single-modality training
- contains the encoder and decoder used for Sentinel-1, Sentinel-2, or other prepared band stacks
- provides the Lightning training module for reconstruction-based pretraining

<img src="./figures/ModalityEncoder.png" alt="Modality-specific autoencoder" width="450">

### 2. Fusion model

Use [`model_fusion.py`](model_fusion.py).

Purpose:
- combines pretrained modality-specific encoders and decoders in the second training stage
- fuses Sentinel-1 and Sentinel-2 representations in a shared latent space
- reconstructs both modalities from the fused representation

<img src="./figures/FusionModel.png" alt="Fusion model" width="350">

### 3. Shared model blocks

Use [`model_blocks.py`](model_blocks.py).

Purpose:
- provides reusable encoder and decoder blocks
- contains the dimensionality reduction and upscaling modules used by the autoencoders
- these blocks are used by both the modality-specific and fusion models.

### 4. Attention components

Use [`attention.py`](attention.py).

Purpose:
- hared attention utilities used across the model definitions
- defines channel and spatial attention blocks
- provides the temporal positional embedding used in the Transformer-based models

### 5. Loss functions

Use [`loss.py`](loss.py).

Purpose:
- defines context-aware reconstruction losses used during training
- combines weighted MAE, SSIM, and SAM terms
- supports spatial and temporal weighting

<img src="./figures/context-aware.png" alt="Context-aware loss concept" width="150">

