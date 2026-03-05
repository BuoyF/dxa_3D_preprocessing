import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.ukb_dataset import create_ukb_dataloaders


# -------------------------
# Models
# -------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = F.relu(out + identity, inplace=True)
        return out


class DXAResNetEncoder(nn.Module):
    """Light ResNet-ish DXA encoder (B,1,H,W)->(B,latent_dim)"""
    def __init__(self, latent_dim=512, base=32, drop=0.10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            BasicBlock(base, base, 1, drop),
            BasicBlock(base, base, 1, drop),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(base, base * 2, 2, drop),
            BasicBlock(base * 2, base * 2, 1, drop),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(base * 2, base * 4, 2, drop),
            BasicBlock(base * 4, base * 4, 1, drop),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(base * 4, base * 8, 2, drop),
            BasicBlock(base * 8, base * 8, 1, drop),
        )
        # Keep spatial resolution for regional composition
        self.pool = nn.AdaptiveAvgPool2d((8, 3))
        # base*8=256 -> 256*8*3=6144
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(6144, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(1024, latent_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.fc(x)


class AnthropometricEncoder(nn.Module):
    """Encode anthropometrics (B,5)->(B,64)"""
    def __init__(self, n_features=5, output_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class CompositionDecoder(nn.Module):
    """Predict composition (B,latent)->(B,M)"""
    def __init__(self, latent_dim=512, n_outputs=130):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(256, n_outputs),
        )

    def forward(self, x):
        return self.fc(x)


class DXAPretrainModel(nn.Module):
    def __init__(self, latent_dim=512, n_composition_outputs=130, backbone_drop=0.10):
        super().__init__()
        self.dxa_encoder = DXAResNetEncoder(latent_dim=latent_dim, base=32, drop=backbone_drop)
        self.anthro_encoder = AnthropometricEncoder(n_features=10, output_dim=64)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + 64, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = CompositionDecoder(latent_dim=latent_dim, n_outputs=n_composition_outputs)

    def forward(self, dxa_image, anthro_normed):
        z_img = self.dxa_encoder(dxa_image)
        z_a = self.anthro_encoder(anthro_normed)
        # Modality dropout to force image learning
        if self.training and torch.rand(1).item() < 0.25:
            z_a = torch.zeros_like(z_a)
        z = self.fusion(torch.cat([z_img, z_a], dim=1))
        y = self.decoder(z)
        return y, z


# -------------------------
# Loss + metrics
# -------------------------

def masked_smooth_l1(pred, target, mask, beta=1.0):
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


@torch.no_grad()
def masked_mae(pred, target, mask):
    err = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return err.sum() / denom


@torch.no_grad()
def per_target_mae_z(pred, target, mask):
    err = (pred - target).abs() * mask
    denom = mask.sum(dim=0).clamp_min(1.0)
    return (err.sum(dim=0) / denom)


# -------------------------
# EMA
# -------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    def apply(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[n] = p.detach().clone()
            p.data.copy_(self.shadow[n].data)

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.backup[n].data)
        self.backup = {}


# -------------------------
# Anthro normalization
# -------------------------

@torch.no_grad()
def compute_anthro_norm(train_loader, device):
    """
    Compute mean/std of anthropometric features from the entire matched dataset.
    This ensures ethnicity one-hot vectors get proper statistics.
    Optimized to avoid expensive DICOM loading.
    """
    try:
        # Try to get dataset from DataLoader
        base_ds = train_loader.dataset
        if hasattr(base_ds, 'dataset'):  # Subset case
            base_ds = base_ds.dataset
        
        # Compute directly from matched_df, avoiding slow DICOM loading
        if hasattr(base_ds, 'matched_df'):
            matched_df = base_ds.matched_df
            
            xs = []
            for idx in range(len(matched_df)):
                try:
                    comp_row = matched_df.iloc[idx]
                    
                    # Replicate the anthropometric extraction logic with proper defaults
                    def _get(val, default=np.nan):
                        try:
                            f = float(val)
                            return f if np.isfinite(f) else default
                        except:
                            return default
                    
                    height_cm = _get(comp_row.get("UKB Field 50 (i0)", np.nan), 170.0)
                    weight_kg = _get(comp_row.get("UKB Field 21002 (i0)", np.nan), 70.0)
                    age_val = _get(comp_row.get("p21022", np.nan), 50.0)
                    
                    # Sex encoding
                    sex_raw = comp_row.get("p31", np.nan)
                    if pd.notna(sex_raw):
                        if isinstance(sex_raw, str):
                            sex_encoded = 1.0 if sex_raw.upper().startswith("M") else 0.0
                        else:
                            try:
                                sex_encoded = 1.0 if float(sex_raw) > 0.5 else 0.0
                            except:
                                sex_encoded = 0.5
                    else:
                        sex_encoded = 0.5
                    
                    # BMI
                    h_m = height_cm / 100.0
                    bmi_val = weight_kg / (h_m * h_m) if h_m > 0 else 25.0
                    
                    # Ethnicity one-hot (text-based)
                    eth_raw = None
                    for col in ["UKB Field 21000 (i3)", "UKB Field 21000 (i0)"]:
                        if col in matched_df.columns and pd.notna(comp_row[col]):
                            eth_raw = str(comp_row[col]).lower()
                            break
                    
                    eth_one_hot = [0.0, 0.0, 0.0, 0.0, 0.0]
                    if eth_raw is not None:
                        if "white" in eth_raw or "british" in eth_raw:
                            eth_one_hot[0] = 1.0
                        elif "black" in eth_raw or "caribbean" in eth_raw or "african" in eth_raw:
                            eth_one_hot[1] = 1.0
                        elif "asian" in eth_raw or "indian" in eth_raw or "pakistani" in eth_raw or "chinese" in eth_raw:
                            eth_one_hot[2] = 1.0
                        elif "mixed" in eth_raw:
                            eth_one_hot[3] = 1.0
                        else:
                            eth_one_hot[4] = 1.0
                    else:
                        eth_one_hot[4] = 1.0
                    
                    anthro = [height_cm, weight_kg, age_val, sex_encoded, bmi_val] + eth_one_hot
                    xs.append(torch.tensor(anthro, dtype=torch.float32))
                    
                except Exception as e:
                    if idx % 1000 == 0:
                        print(f"  Warning at sample {idx}: {e}")
                    continue
            
            if xs:
                x = torch.stack(xs).to(device)
                mean = x.mean(dim=0)
                std = x.std(dim=0).clamp_min(1e-6)
                print(f"  Computed stats from {len(xs)}/{len(matched_df)} dataset samples")
                return mean, std
    except Exception as e:
        print(f"  Warning: Could not use matched_df: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback: iterate through DataLoader
    print("  Falling back to DataLoader iteration...")
    xs = []
    for i, batch in enumerate(train_loader):
        xs.append(batch["anthropometric"].float())
        if i >= 50:
            break
    x = torch.cat(xs, dim=0).to(device)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    return mean, std


# -------------------------
# Validation + reporting
# -------------------------

@torch.no_grad()
def validate(model, val_loader, device, anthro_mean, anthro_std, huber_beta,
             comp_fields=None, comp_std=None, out_dir: Path = None):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0

    per_sum = None
    per_den = None

    for batch in val_loader:
        dxa = batch["dxa_image"].to(device, non_blocking=True)
        y = batch["composition"].to(device, non_blocking=True)
        m = batch.get("composition_mask", torch.ones_like(y)).to(device, non_blocking=True)
        anthro = batch["anthropometric"].to(device, non_blocking=True)
        anthro = (anthro - anthro_mean) / anthro_std

        pred, _ = model(dxa, anthro)
        loss = masked_smooth_l1(pred, y, m, beta=huber_beta)
        mae = masked_mae(pred, y, m)

        total_loss += float(loss.item())
        total_mae += float(mae.item())

        err = (pred - y).abs() * m
        den = m.sum(dim=0)

        if per_sum is None:
            per_sum = err.sum(dim=0).detach().cpu()
            per_den = den.detach().cpu()
        else:
            per_sum += err.sum(dim=0).detach().cpu()
            per_den += den.detach().cpu()

    val_loss = total_loss / len(val_loader)
    val_mae = total_mae / len(val_loader)

    metrics = {"val_loss": val_loss, "val_mae_z": val_mae}

    # Per-target reporting (paper-friendly)
    if out_dir is not None and comp_fields is not None:
        per_mae_z = (per_sum / per_den.clamp_min(1.0)).numpy()
        metrics["per_target_mae_z"] = per_mae_z.tolist()

        # Approx real-unit MAE via std * MAE(z)
        if comp_std is not None:
            comp_std = np.asarray(comp_std, dtype=np.float32)
            per_mae_real = (per_mae_z * comp_std).astype(np.float32)
            metrics["per_target_mae_real"] = per_mae_real.tolist()

        out_dir.mkdir(parents=True, exist_ok=True)
        # CSV
        with open(out_dir / "val_per_target.csv", "w", newline="") as f:
            w = csv.writer(f)
            header = ["idx", "field", "mae_z"]
            if "per_target_mae_real" in metrics:
                header.append("mae_real(approx)")
            w.writerow(header)
            for i, name in enumerate(comp_fields):
                row = [i, name, float(per_mae_z[i])]
                if "per_target_mae_real" in metrics:
                    row.append(float(metrics["per_target_mae_real"][i]))
                w.writerow(row)

        # summary JSON (top/worst)
        order = np.argsort(per_mae_z)
        best = order[:10].tolist()
        worst = order[-10:][::-1].tolist()
        summary = {
            "val_loss": val_loss,
            "val_mae_z": val_mae,
            "best_targets": [{"idx": i, "field": comp_fields[i], "mae_z": float(per_mae_z[i])} for i in best],
            "worst_targets": [{"idx": i, "field": comp_fields[i], "mae_z": float(per_mae_z[i])} for i in worst],
        }
        (out_dir / "val_summary.json").write_text(json.dumps(summary, indent=2))

    return metrics


# -------------------------
# Train loop
# -------------------------

def train_epoch(model, ema, train_loader, optimizer, scheduler, device,
                anthro_mean, anthro_std, huber_beta, use_amp=True,
                grad_clip=1.0, grad_accum=1, epoch=1):
    model.train()

    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    total_loss = 0.0
    total_mae = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar, start=1):
        dxa = batch["dxa_image"].to(device, non_blocking=True)
        y = batch["composition"].to(device, non_blocking=True)
        m = batch.get("composition_mask", torch.ones_like(y)).to(device, non_blocking=True)
        anthro = batch["anthropometric"].to(device, non_blocking=True)
        anthro = (anthro - anthro_mean) / anthro_std

        with torch.amp.autocast(autocast_device, enabled=use_amp):
            pred, _ = model(dxa, anthro)
            loss = masked_smooth_l1(pred, y, m, beta=huber_beta)
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if step % grad_accum == 0:
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            ema.update(model)

        mae = masked_mae(pred.detach(), y, m).item()
        total_loss += float(loss.item() * grad_accum)
        total_mae += float(mae)
        pbar.set_postfix({"loss": f"{(loss.item()*grad_accum):.3f}", "mae(z)": f"{mae:.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.1e}"})

    return total_loss / len(train_loader), total_mae / len(train_loader)


# -------------------------
# Main
# -------------------------

def pretrain(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"UKB DXA Pretraining — CUDA ({props.name}, {props.total_memory / (1024**3):.1f} GB)")
    else:
        print("UKB DXA Pretraining — CPU")

    print("\nCreating dataloaders...")
    train_loader, val_loader = create_ukb_dataloaders(
        dicom_dir=args.dicom_dir,
        composition_csv=args.composition_csv,
        instance=args.instance,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        target_size=(320, 864),
        augment=args.augment,
        # DXA physics preservation
        augment_params={
            "rotation": 5.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "gaussian_noise": 0.01,
        } if args.augment else None,
        min_non_missing_ratio=args.min_non_missing_ratio,
        impute_missing_with_mean=True,
        verbose=False,
    )

    sample_batch = next(iter(train_loader))
    n_outputs = sample_batch["composition"].shape[1]
    print(f"Outputs: {n_outputs} targets")

    model = DXAPretrainModel(
        latent_dim=args.latent_dim,
        n_composition_outputs=n_outputs,
        backbone_drop=args.backbone_drop,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # OneCycleLR: no restarts, no nonsense, strong convergence for convnets.
    steps_per_epoch = len(train_loader) // max(1, args.grad_accum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.10,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    ema = EMA(model, decay=args.ema_decay)

    # Pull target metadata for reporting
    base_ds = train_loader.dataset.dataset  # Subset.dataset
    comp_fields = getattr(base_ds, "composition_fields", None)
    comp_std = getattr(base_ds, "composition_std", None)

    # Anthro norm from train split
    print("Computing anthropometric normalization...")
    anthro_mean, anthro_std = compute_anthro_norm(train_loader, device)
    print("Anthro norm:", anthro_mean.detach().cpu().numpy().round(3), anthro_std.detach().cpu().numpy().round(3))

    print("\n========")
    print("TRAINING")
    print("========")

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_mae = train_epoch(
            model=model,
            ema=ema,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            anthro_mean=anthro_mean,
            anthro_std=anthro_std,
            huber_beta=args.huber_beta,
            use_amp=args.amp,
            grad_clip=args.grad_clip,
            grad_accum=args.grad_accum,
            epoch=epoch,
        )

        # EMA validation
        ema.apply(model)
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            anthro_mean=anthro_mean,
            anthro_std=anthro_std,
            huber_beta=args.huber_beta,
            comp_fields=comp_fields,
            comp_std=comp_std,
            out_dir=args.output_dir,
        )
        ema.restore(model)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} — "
            f"train {train_loss:.4f}, val {val_metrics['val_loss']:.4f}, "
            f"val_MAE(z) {val_metrics['val_mae_z']:.4f}, lr {lr_now:.2e}, "
            f"time {dt/60:.1f}m"
        )

        if val_metrics["val_loss"] < best_val - 1e-6:
            best_val = val_metrics["val_loss"]
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_shadow": ema.shadow,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": float(val_metrics["val_loss"]),
                "val_mae_z": float(val_metrics["val_mae_z"]),
                "n_composition_outputs": int(n_outputs),
                "latent_dim": int(args.latent_dim),
                "anthro_mean": anthro_mean.detach().cpu().numpy(),
                "anthro_std": anthro_std.detach().cpu().numpy(),
                "composition_fields": comp_fields,
                "composition_std": comp_std,
                "args": vars(args),
            }

            save_path = args.output_dir / "best_pretrained_model.pth"
            torch.save(checkpoint, save_path)
            print(f"Saved best model: {save_path} (val {best_val:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}. Best val {best_val:.4f}")
            break

    print(f"\nDone. Best val: {best_val:.4f}. Model: {args.output_dir / 'best_pretrained_model.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain on UK Biobank DXA data (masked targets, stable schedule)")

    # Data
    parser.add_argument("--dicom-dir", type=str, default=r"D:\uk\extracted")
    parser.add_argument("--composition-csv", type=str, default=r"D:\uk\string_participant_instance3_readable.csv",
                        help="Path to composition CSV (readable version with human-friendly column names)")
    parser.add_argument("--instance", type=int, default=3)

    # Split / loader
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--backbone-drop", type=float, default=0.10)

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)  # OneCycle likes a slightly higher peak LR
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=12)

    # Loss
    parser.add_argument("--huber-beta", type=float, default=1.0)

    # Dataset filter
    parser.add_argument("--min-non-missing-ratio", type=float, default=0.10)

    # Augment
    parser.add_argument("--augment", action="store_true", default=True)

    # Stability / speed
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1)

    # EMA
    parser.add_argument("--ema-decay", type=float, default=0.999)

    # Output
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/pretraining"))

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pretrain(args)