import random
import sys
from pathlib import Path

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.dataloader import HDF5Dataset
from model.model_s1_s2 import TransformerAE

torch.set_float32_matmul_precision("medium")

SEED = 42
DEVICE_ID = 1
BATCH_SIZE = 16
NUM_WORKERS = 24
MAX_EPOCHS = 500

# Set this to "s1"or "s2" depending on the modality-specific stage.
MODALITY = "s2"

MODALITY_CONFIG = {
    "s1": {
        "channels": 2,
        "dbottleneck": 2,
        "num_reduced_tokens": 7,
    },
    "s2": {
        "channels": 10,
        "dbottleneck": 9,
        "num_reduced_tokens": 7,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_paths(modality: str) -> tuple[Path, Path, Path]:
    train_h5 = ROOT / f"train_{modality}.h5"
    val_h5 = ROOT / f"val_{modality}.h5"
    ckpt_dir = ROOT / "checkpoints" / modality
    return train_h5, val_h5, ckpt_dir


def main() -> None:
    if MODALITY not in MODALITY_CONFIG:
        raise ValueError(f"Unknown MODALITY: {MODALITY}")

    set_seed(SEED)
    cfg = MODALITY_CONFIG[MODALITY]
    train_h5, val_h5, ckpt_dir = build_paths(MODALITY)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = HDF5Dataset(str(train_h5))
    val_dataset = HDF5Dataset(str(val_h5))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = TransformerAE(
        dbottleneck=cfg["dbottleneck"],
        channels=cfg["channels"],
        num_reduced_tokens=cfg["num_reduced_tokens"],
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(ckpt_dir),
        filename=f"ae-{cfg['dbottleneck']}" + "-{epoch:02d}-{val_loss:.3e}",
        save_top_k=5,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=16,
        mode="min",
        verbose=True,
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        devices=[DEVICE_ID],
        accelerator="gpu",
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    with torch.autograd.set_detect_anomaly(True):
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
