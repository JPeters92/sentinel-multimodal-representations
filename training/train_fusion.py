import random
import sys
from pathlib import Path
import re

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.dataloader import HDF5Dataset
from model.model_fusion import FusedS1S2, load_enc_dec_from_ae_ckpt

torch.set_float32_matmul_precision("medium")

SEED = 42
DEVICE_ID = 1
BATCH_SIZE = 16
NUM_WORKERS = 24
MAX_EPOCHS = 500

TRAIN_H5 = ROOT / "train_s1_s2.h5"
VAL_H5 = ROOT / "val_s1_s2.h5"
CHECKPOINT_DIR = ROOT / "checkpoints" / "fusion" / "s1_s2"

DBOTTLENECK_FUSION = 7
MODALITY_CKPT_DIRS = {
    "s1": ROOT / "checkpoints" / "modality" / "s1",
    "s2": ROOT / "checkpoints" / "modality" / "s2",
}
CKPT_PATTERN = re.compile(r"val_loss=([0-9.]+e[+-]?\d+)")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_best_checkpoint(ckpt_dir: Path) -> Path:
    candidates = sorted(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    scored: list[tuple[float, Path]] = []
    for path in candidates:
        match = CKPT_PATTERN.search(path.name)
        if match is not None:
            scored.append((float(match.group(1)), path))

    if scored:
        return min(scored, key=lambda item: item[0])[1]
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_modality_checkpoint_config(ckpt_path: Path) -> dict[str, int]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})
    return {
        "channels": int(hparams["channels"]),
        "dbottleneck": int(hparams["dbottleneck"]),
        "num_reduced_tokens": int(hparams["num_reduced_tokens"]),
    }


def main() -> None:
    set_seed(SEED)
    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    s1_ckpt = find_best_checkpoint(MODALITY_CKPT_DIRS["s1"])
    s2_ckpt = find_best_checkpoint(MODALITY_CKPT_DIRS["s2"])
    s1_cfg = read_modality_checkpoint_config(s1_ckpt)
    s2_cfg = read_modality_checkpoint_config(s2_ckpt)

    train_dataset = HDF5Dataset(str(TRAIN_H5), s1_s2=True)
    val_dataset = HDF5Dataset(str(VAL_H5), s1_s2=True)

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

    enc_s1, dec_s1 = load_enc_dec_from_ae_ckpt(
        device=device,
        ckpt_path=str(s1_ckpt),
        channels=s1_cfg["channels"],
        dbottleneck=s1_cfg["dbottleneck"],
        num_reduced_tokens=s1_cfg["num_reduced_tokens"],
    )
    enc_s2, dec_s2 = load_enc_dec_from_ae_ckpt(
        device=device,
        ckpt_path=str(s2_ckpt),
        channels=s2_cfg["channels"],
        dbottleneck=s2_cfg["dbottleneck"],
        num_reduced_tokens=s2_cfg["num_reduced_tokens"],
    )

    model = FusedS1S2(
        enc_s1=enc_s1,
        dec_s1=dec_s1,
        enc_s2=enc_s2,
        dec_s2=dec_s2,
        dbottleneck_s1=s1_cfg["dbottleneck"],
        dbottleneck_s2=s2_cfg["dbottleneck"],
        dbottleneck=DBOTTLENECK_FUSION,
        freeze_encoders=False,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(CHECKPOINT_DIR),
        filename="fuse_model",
        save_top_k=1,
        save_last=False,
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
