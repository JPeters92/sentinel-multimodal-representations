import math
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

try:
    from model.loss import WeightedMaskedLoss
    from model.attention import TemporalPositionalEmbedding
    from model.model_s1_s2 import TransformerAE
    from model.model_blocks import *
except:
    from loss import WeightedMaskedLoss
    from attention import TemporalPositionalEmbedding
    from model_s1_s2 import TransformerAE
    from model_blocks import *


def load_enc_dec_from_ae_ckpt(
    device: torch.device,
    ckpt_path: str,
    *,
    channels: int,
    dbottleneck: int,
    num_reduced_tokens: int = 6,
) -> tuple[nn.Module, nn.Module]:
    """
    Lädt einen vollständigen TransformerAE aus dem Lightning-Checkpoint
    und gibt dessen encoder/decoder-Module zurück (mit geladenen Gewichten).
    """
    ae = TransformerAE(dbottleneck=dbottleneck, channels=channels, num_reduced_tokens=num_reduced_tokens)
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=device)
        ae.load_state_dict(checkpoint['state_dict'])
        ae.to(device)
        ae.eval()
    # Encoder/Decoder als eigenständige Module referenzieren (mit weights)
    enc = ae.encoder
    dec = ae.decoder
    return enc, dec

def compute_cumulative_positions(time_gaps):
    # Initialize cumulative positions with zeros for each sample in the batch
    batch_size, frames_minus_one = time_gaps.size()
    cumulative_positions = torch.zeros((batch_size, frames_minus_one + 1), dtype=torch.long, device=time_gaps.device)

    # Fill cumulative positions by computing the cumulative sum for each batch sample
    cumulative_positions[:, 1:] = torch.cumsum(time_gaps, dim=1)

    return cumulative_positions

class FusedS1S2(pl.LightningModule):
    """
    Fuse pretrained S1 and S2 encoders with a 2-token Transformer. Reconstruct both modalities.

    forward inputs:
        x_s1, gaps_s1, mask_s1
        x_s2, gaps_s2, mask_s2
        pair_time_delta_days  (B,) or (B,1)  # days offset between S1 and S2

    forward outputs:
        y_s1: (B, T1, 2, 15, 15)
        y_s2: (B, T2, 10, 15, 15)
        z_fused: (B, d_fuse)
    """
    def __init__(
        self,
        enc_s1: nn.Module, dec_s1: nn.Module,         # pretrained S1 encoder/decoder
        enc_s2: nn.Module, dec_s2: nn.Module,         # pretrained S2 encoder/decoder
        dbottleneck_s1: int = 7,
        dbottleneck_s2: int = 7,
        dbottleneck: int = 7,
        d_fuse: int = 128,                            # shared fusion dim
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        max_position: int = 350,                      # for pair positions [0, Δt]
        learning_rate: float = 1e-4,
        freeze_encoders: bool = True,
        loss_fn_s1: nn.Module | None = None,
        loss_fn_s2: nn.Module | None = None,
        loss_fn: nn.Module | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['enc_s1','dec_s1','enc_s2','dec_s2','loss_fn_s1', 'loss_fn_s2'])

        # plug in pretrained parts
        self.enc_s1 = enc_s1
        self.enc_s2 = enc_s2
        self.dec_s1 = dec_s1
        self.dec_s2 = dec_s2

        # optionally freeze encoders first
        if freeze_encoders:
            for p in self.enc_s1.parameters(): p.requires_grad = False
            for p in self.dec_s1.parameters(): p.requires_grad = False
            for p in self.enc_s2.parameters(): p.requires_grad = False
            for p in self.dec_s2.parameters(): p.requires_grad = False

        # project each modality bottleneck to common fusion dim
        self.proj_s1 = nn.Linear(dbottleneck_s1, d_fuse)
        self.proj_s2 = nn.Linear(dbottleneck_s2, d_fuse)

        # modality/type embedding (2 tokens: 0=S1, 1=S2)
        self.type_embed = nn.Embedding(2, d_fuse)

        # temporal positional embeddings for 2-token stream
        self.pair_pos_emb = TemporalPositionalEmbedding(d_model=d_fuse, max_position=max_position)

        # tiny transformer over 2 tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_fuse, nhead=nhead, dim_feedforward=dim_ff, dropout=0.1, batch_first=True, norm_first=True
        )
        self.fuser = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # pooled fused rep
        self.pool = nn.AdaptiveAvgPool1d(1)  # mean over token dim after transpose
        self.norm_out = nn.LayerNorm(d_fuse)

        # adapters to route fused rep back to each decoder bottleneck
        self.head_s1 = nn.Linear(dbottleneck, dbottleneck_s1)
        self.head_s2 = nn.Linear(dbottleneck, dbottleneck_s2)

        self.loss_fn_s1 = loss_fn_s1 if loss_fn_s1 is not None else WeightedMaskedLoss(lambda_mae=1., lambda_sam=0., lambda_ssim=0., center_mae=1.5)
        self.loss_fn_s2 = loss_fn_s2 if loss_fn_s2 is not None else WeightedMaskedLoss(lambda_mae=0.665, lambda_sam=0.315, lambda_ssim=0.02, center_mae=1.75) #0.018
        #self.loss_fn_s2 = loss_fn_s2 if loss_fn_s2 is not None else WeightedMaskedLoss(lambda_mae=1., lambda_sam=0., lambda_ssim=0.00, center_mae=1.75)
        #self.loss_fn_s2 = loss_fn_s2 if loss_fn_s2 is not None else WeightedMaskedLoss(lambda_mae=0.75, lambda_sam=0.24, lambda_ssim=0.01, center_mae=1.7)

        #self.loss_fn = loss_fn if loss_fn is not None else WeightedMaskedLoss(lambda_mae=0., lambda_sam=0.9, lambda_ssim=0.1)
        self.learning_rate = learning_rate

        self.bottleneck_reducer = nn.Sequential(
            #nn.Linear(dbottleneck_s1 + dbottleneck_s2, dbottleneck),
            nn.Linear(d_fuse, dbottleneck),
            nn.Softplus(beta=1, threshold=20)
        )

        self.bottleneck_expander = nn.Sequential(
            nn.Linear(dbottleneck, d_fuse),
            nn.Softplus(beta=1, threshold=20)
        )

    def fuse_tokens(self, z1: torch.Tensor, z2: torch.Tensor, time_gaps: torch.Tensor) -> torch.Tensor:
        """
        z1: (B, d_b1)  from enc_s1
        z2: (B, d_b2)  from enc_s2
        returns fused: (B, d_fuse)
        """
        t1 = self.proj_s1(z1)  # (B, d_fuse)
        t2 = self.proj_s2(z2)  # (B, d_fuse)
        tokens = torch.stack([t1, t2], dim=1)  # (B, 2, d_fuse)

        # add modality/type embeddings
        type_ids = torch.tensor([0, 1], device=tokens.device).unsqueeze(0).expand(tokens.size(0), -1)  # (B, 2)
        tokens = tokens + self.type_embed(type_ids)

        # add pair positions [0, Δt]
        cumulative_positions = compute_cumulative_positions(time_gaps)  # Shape: (batch_size, frames)

        pos_emb = self.pair_pos_emb(cumulative_positions)  # Shape: (batch_size, frames, d2)
        pos_emb = pos_emb / torch.sqrt(torch.tensor(pos_emb.size(-1), dtype=torch.float))
        tokens = tokens + pos_emb

        # fuse with transformer

        h = self.fuser(tokens)  # (B, 2, d_fuse)

        # POOL over the token dimension to get a single fused vector
        h = self.pool(h.transpose(1, 2)).squeeze(-1)  # (B, d_fuse)
        h = self.norm_out(h)
        h = self.bottleneck_reducer(h)

        # (B, d_fuse)
        return h                             # (B, d_fuse)

    def forward(self, batch):
        """
        batch should be a dict or tuple; here we support a tuple in order:
          x_s1, gaps_s1, mask_s1, x_s2, gaps_s2, mask_s2, pair_time_delta_days
        """
        (x_s1, x_s2, gaps_s1, gaps_s2, pair_time_delta_days) = batch

        # encode each modality to its bottleneck (context-aware, compressing T)
        z1 = self.enc_s1(x_s1, gaps_s1)   # (B, dbottleneck_s1)
        z2 = self.enc_s2(x_s2, gaps_s2)   # (B, dbottleneck_s2)

        # fuse with 2-token transformer using [0, Δt] positions
        zf = self.fuse_tokens(z1, z2, pair_time_delta_days)    # (B, d_fuse)



        # map back to each modality bottleneck and decode full sequences
        b1 = self.head_s1(zf)                                   # (B, dbottleneck_s1)
        b2 = self.head_s2(zf)                                   # (B, dbottleneck_s2)

        y_s1 = self.dec_s1(b1, gaps_s1)                         # (B, T1, 2, 15, 15)
        y_s2 = self.dec_s2(b2, gaps_s2)                         # (B, T2,10, 15, 15)
        return y_s1, y_s2, zf

    # -------- Lightning steps --------
    def training_step(self, batch, batch_idx):
        #x_s1, gaps_s1, mask_s1, x_s2, gaps_s2, mask_s2, s1_s2_delta = batch
        (x, mask, gaps_s1, gaps_s2, s1_s2_delta) = batch
        x_s2 = x[:, :, :10, :, :]
        x_s1 = x[:, :, 10:, :, :]

        mask_s2 = mask[:, :, :10, :, :]
        mask_s1 = mask[:, :, 10:, :, :]

        model_input = (x_s1, x_s2, gaps_s1, gaps_s2, s1_s2_delta)

        y_s1, y_s2, zf = self(model_input)

        y_all = torch.cat([y_s2, y_s1], dim=2)
        x_all = torch.cat([x_s2, x_s1], dim=2)

        # reconstruction losses (your WeightedMaskedLoss works on (pred, target, mask))
        total_s1, mae_s1, _, _, center_s1 = self.loss_fn_s1(y_s1, x_s1, mask_s1)
        total_s2, mae_s2, _, _, center_s2 = self.loss_fn_s2(y_s2, x_s2, mask_s2)
        total = 0.5 * total_s1 + 0.5 * total_s2 #+ 0.1 * total

        self.log_dict({
            "t_total": total,
            "t_s1_total": total_s1, "t_s2_total": total_s2,
            "t_s1_c": center_s1, "t_s2_c": center_s2,
        }, prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,)
        # optional LR log
        opt = self.optimizers()
        if opt and len(opt.param_groups) > 0:
            self.log("lr", opt.param_groups[0]["lr"], prog_bar=True, on_step=True)
        return total

    def validation_step(self, batch, batch_idx):

        (x, mask, gaps_s1, gaps_s2, s1_s2_delta) = batch
        x_s2 = x[:, :, :10, :, :]
        x_s1 = x[:, :, 10:, :, :]

        mask_s2 = mask[:, :, :10, :, :]
        mask_s1 = mask[:, :, 10:, :, :]
        model_input = (x_s1, x_s2, gaps_s1, gaps_s2, s1_s2_delta)
        y_s1, y_s2, zf = self(model_input)

        y_all = torch.cat([y_s2, y_s1], dim=2)
        x_all = torch.cat([x_s2, x_s1], dim=2)

        total_s1, mae_s1, ssim_s1, sam_s1, center_s1 = self.loss_fn_s1(y_s1, x_s1, mask_s1, val=True)
        total_s2, mae_s2, ssim_s2, sam_s2, center_s2 = self.loss_fn_s2(y_s2, x_s2, mask_s2, val=True)
        val_center = (center_s1 + center_s2) / 2.

        self.log_dict({
            "l_total": total_s1 + total_s2,
            "v_s1_mae": mae_s1, "v_s2_mae": mae_s2,
            "v_ssim": ssim_s2,
            "v_sam": sam_s2,
            "v_s1_c": center_s1, "v_s2_c": center_s2,

            "val_loss": val_center,
        }, prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        return val_center

    def test_step(self, batch, batch_idx):
        y_s1, y_s2, _ = self(batch)
        (x_s1, _, mask_s1, x_s2, _, mask_s2, _) = batch
        total_s1, mae_s1, ssim_s1, sam_s1, center_s1 = self.loss_fn_s1(y_s1, x_s1, mask_s1, val=True)
        total_s2, mae_s2, ssim_s2, sam_s2, center_s2 = self.loss_fn_s1(y_s2, x_s2, mask_s2, val=True)
        self.log_dict({
            "test_total": total_s1 + total_s2,
            "test_s1_total": total_s1, "test_s2_total": total_s2,
            "test_s1_mae": mae_s1, "test_s2_mae": mae_s2,
            "test_s1_ssim": ssim_s1, "test_s2_ssim": ssim_s2,
            "test_s1_sam": sam_s1, "test_s2_sam": sam_s2,
            "test_s1_center": center_s1, "test_s2_center": center_s2,
        }, prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        return total_s1 + total_s2

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1.5e-4)

        warmup = {
            "scheduler": LinearLR(opt, start_factor=0.001, total_iters=30000),

            "interval": "step",
            "frequency": 1,
        }
        plateau = {
            "scheduler": ReduceLROnPlateau(
                #opt, mode="min", factor=0.35, patience=5, verbose=False,
                opt, mode="min", factor=0.3, patience=9, verbose=False,
                #min_lr=1.5e-6, threshold=1.5e-6, threshold_mode="abs"
                min_lr=3.5e-7, threshold=3.5e-7, threshold_mode="abs"
            ),
            "interval": "epoch",
            "monitor": "val_loss",
        }
        return [opt], [warmup, plateau]


def main():
    # Debug-Hyperparameter
    batch_size = 2
    frames_s1 = 11   # Anzahl Zeitpunkte Sentinel-1
    frames_s2 = 11  # Anzahl Zeitpunkte Sentinel-2
    chans_s1 = 2
    chans_s2 = 10
    dbottleneck_s1 = 2
    dbottleneck_s2 = 9

    s2_ckpt = '../checkpoints//s2/ae-9-epoch=154-val_loss=3.686e-03.ckpt'
    s1_ckpt = '../checkpoints//s1/ae-2-epoch=68-val_loss=6.832e-04.ckpt'
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    enc_s1, dec_s1 = load_enc_dec_from_ae_ckpt(
        device=device,
        ckpt_path=s1_ckpt,
        channels=chans_s1,
        dbottleneck=dbottleneck_s1,
        num_reduced_tokens=7
    )
    enc_s2, dec_s2 = load_enc_dec_from_ae_ckpt(
        device=device,
        ckpt_path=s2_ckpt,
        channels=chans_s2,
        dbottleneck=dbottleneck_s2,
        num_reduced_tokens=6
    )

    # Modell instanziieren
    model = FusedS1S2(
        enc_s1=enc_s1, dec_s1=dec_s1,
        enc_s2=enc_s2, dec_s2=dec_s2,
        dbottleneck_s1=dbottleneck_s1,
        dbottleneck_s2=dbottleneck_s2,
        freeze_encoders=False
    ).to(device)

    # Dummy Inputs bauen
    x_s1 = torch.randn(batch_size, frames_s1, chans_s1, 15, 15, device=device)
    x_s2 = torch.randn(batch_size, frames_s2, chans_s2, 15, 15, device=device)
    gaps_s1 = torch.randint(1, 5, (batch_size, frames_s1 - 1), device=device, dtype=torch.long)
    gaps_s2 = torch.randint(1, 5, (batch_size, frames_s2 - 1), device=device, dtype=torch.long)
    mask_s1 = torch.ones_like(x_s1, dtype=torch.bool, device=device)
    mask_s2 = torch.ones_like(x_s2, dtype=torch.bool, device=device)
    pair_time_delta_days = torch.randint(0, 30, (batch_size, 1), device=device, dtype=torch.long)

    # Forward-Pass
    model.eval()
    with torch.no_grad():
        y_s1, y_s2, zf = model((x_s1, gaps_s1, x_s2, gaps_s2, pair_time_delta_days))

    print("============== Debug ==============")
    print("x_s1:", x_s1.shape, "x_s2:", x_s2.shape)
    print("gaps_s1:", gaps_s1.shape, "gaps_s2:", gaps_s2.shape)
    print("y_s1:", y_s1.shape, "y_s2:", y_s2.shape)
    print("zf (fused):", zf.shape)

    # Loss testen
    loss_vals = model.loss_fn_s1(y_s1, x_s1, mask_s1)
    print("Loss output (s1):", loss_vals)
    loss_vals = model.loss_fn_s1(y_s2, x_s2, mask_s2)
    print("Loss output (s2):", loss_vals)

if __name__ == "__main__":
    main()
