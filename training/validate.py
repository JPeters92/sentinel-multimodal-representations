import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.dataloader import HDF5Dataset
from model.model_fusion import FusedS1S2, load_enc_dec_from_ae_ckpt
from model.model_s1_s2 import TransformerAE

torch.set_float32_matmul_precision("medium")

MODE = "fusion"  # "s1", "s2", or "fusion"
DEVICE_ID = 1
BATCH_SIZE = 24
NUM_WORKERS = 8

MODALITY_CKPT_DIRS = {
    "s1": ROOT / "checkpoints" / "modality" / "s1",
    "s2": ROOT / "checkpoints" / "modality" / "s2",
}
FUSION_CKPT_CANDIDATES = [
    ROOT / "checkpoints" / "fusion" / "s1_s2" / "fuse_model.ckpt",
    ROOT / "checkpoints" / "fuse_model.ckpt",
]
CKPT_PATTERN = re.compile(r"val_loss=([0-9.]+e[+-]?\d+)")

TEST_DATASETS = {
    "s1": ROOT / "test_s1.h5",
    "s2": ROOT / "test_s2.h5",
    "fusion": ROOT / "test_s1_s2.h5",
}


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


def resolve_fusion_checkpoint() -> Path:
    for ckpt_path in FUSION_CKPT_CANDIDATES:
        if ckpt_path.exists():
            return ckpt_path
    raise FileNotFoundError("No fusion checkpoint found in expected locations.")


def load_modality_model(mode: str, device: torch.device) -> tuple[torch.nn.Module, Path]:
    ckpt_path = find_best_checkpoint(MODALITY_CKPT_DIRS[mode])
    cfg = read_modality_checkpoint_config(ckpt_path)
    model = TransformerAE(
        dbottleneck=cfg["dbottleneck"],
        channels=cfg["channels"],
        num_reduced_tokens=cfg["num_reduced_tokens"],
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt_path


def load_fusion_model(device: torch.device) -> tuple[torch.nn.Module, Path]:
    s1_ckpt = find_best_checkpoint(MODALITY_CKPT_DIRS["s1"])
    s2_ckpt = find_best_checkpoint(MODALITY_CKPT_DIRS["s2"])
    s1_cfg = read_modality_checkpoint_config(s1_ckpt)
    s2_cfg = read_modality_checkpoint_config(s2_ckpt)
    fusion_ckpt = resolve_fusion_checkpoint()

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
        dbottleneck=7,
        freeze_encoders=False,
    )
    checkpoint = torch.load(fusion_ckpt, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, fusion_ckpt


def evaluate_modality(model: TransformerAE, loader: DataLoader, device: torch.device) -> float:
    total_metric = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating modality", leave=False):
            x, time_gaps, mask = batch
            x = x.to(device)
            time_gaps = time_gaps.to(device)
            mask = mask.to(device)

            recon, _ = model(x, time_gaps=time_gaps)
            _, _, _, _, metric = model.loss_fn(recon, x, mask, val=True)

            batch_size = x.size(0)
            total_metric += metric.item() * batch_size
            total_samples += batch_size

    return total_metric / total_samples


def evaluate_fusion(model: FusedS1S2, loader: DataLoader, device: torch.device) -> float:
    total_metric = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating fusion", leave=False):
            x, mask, gaps_s1, gaps_s2, gaps_c = batch
            x = x.to(device)
            mask = mask.to(device)
            gaps_s1 = gaps_s1.to(device)
            gaps_s2 = gaps_s2.to(device)
            gaps_c = gaps_c.to(device)

            x_s2 = x[:, :, :10, :, :]
            x_s1 = x[:, :, 10:, :, :]
            mask_s2 = mask[:, :, :10, :, :]
            mask_s1 = mask[:, :, 10:, :, :]

            y_s1, y_s2, _ = model((x_s1, x_s2, gaps_s1, gaps_s2, gaps_c))
            _, _, _, _, center_s1 = model.loss_fn_s1(y_s1, x_s1, mask_s1, val=True)
            _, _, _, _, center_s2 = model.loss_fn_s2(y_s2, x_s2, mask_s2, val=True)
            metric = (center_s1 + center_s2) / 2.0

            batch_size = x.size(0)
            total_metric += metric.item() * batch_size
            total_samples += batch_size

    return total_metric / total_samples


def main() -> None:
    if MODE not in TEST_DATASETS:
        raise ValueError(f"Unknown MODE: {MODE}")

    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    dataset_path = TEST_DATASETS[MODE]

    if MODE == "fusion":
        model, ckpt_path = load_fusion_model(device)
        dataset = HDF5Dataset(str(dataset_path), s1_s2=True)
    else:
        model, ckpt_path = load_modality_model(MODE, device)
        dataset = HDF5Dataset(str(dataset_path))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    if MODE == "fusion":
        metric = evaluate_fusion(model, loader, device)
    else:
        metric = evaluate_modality(model, loader, device)

    print(f"Mode: {MODE}")
    print(f"Dataset: {dataset_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Validation metric: {metric:.6f}")


if __name__ == "__main__":
    main()
