"""Final training: MK_UNet_Hybrid_DS at 224x224 with PraNet-style deep supervision.

This variant keeps the FinalDS pipeline structure, but changes the training step
to mirror PraNet's deep-supervision strategy:
  - weighted structure_loss applied to every decoder head
  - per-batch multi-scale training over [0.75, 1.0, 1.25]
  - full trainval for training and test as validation/final evaluation
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import sys
_mkunet_root = str(Path(__file__).resolve().parent.parent / "MK-UNet")
if _mkunet_root not in sys.path:
    sys.path.append(_mkunet_root)

from tn3k_train import (
    EPS, MetricAccumulator, batch_metrics, build_visualization,
    infer_epochs_without_improvement, load_history_csv, numpy_metrics,
    save_checkpoint, seed_everything, seed_worker, setup_logging,
    structure_loss, write_history_csv,
)
from mkunet_hybrid_ds_pranet import MK_UNet_Hybrid_DS_PraNet
from dataset_trainval import TN3KDataset, MASK_THRESHOLD, resolve_tn3k_root

BRANCH_NAMES = ["S3", "S4", "S5", "Sg"]
PRANET_SIZE_RATES = [0.75, 1.0, 1.25]


# ---------------------------------------------------------------------------
# Best ensemble: 8 flip+rot × 3 scales
# ---------------------------------------------------------------------------

def _hflip(x):      return x.flip(-1)
def _vflip(x):      return x.flip(-2)
def _hvflip(x):     return x.flip(-1, -2)
def _rot90(x):       return x.rot90(1, [-2, -1])
def _rot180(x):      return x.rot90(2, [-2, -1])
def _rot270(x):      return x.rot90(3, [-2, -1])
def _hflip_rot90(x): return _rot90(_hflip(x))
def _derot90(x):     return x.rot90(3, [-2, -1])
def _derot180(x):    return x.rot90(2, [-2, -1])
def _derot270(x):    return x.rot90(1, [-2, -1])
def _dehflip_r90(x): return _hflip(_derot90(x))

BEST_AUG_POOL: List[Tuple[Callable, Callable]] = [
    (lambda x: x,   lambda x: x),
    (_hflip,         _hflip),
    (_vflip,         _vflip),
    (_hvflip,        _hvflip),
    (_rot90,         _derot90),
    (_rot180,        _derot180),
    (_rot270,        _derot270),
    (_hflip_rot90,   _dehflip_r90),
]
BEST_SCALES = [0.75, 1.0, 1.25]


def best_ensemble_predict(model: nn.Module, images: torch.Tensor, target_size: int) -> torch.Tensor:
    """8 flip+rot × 3 scales, uniform weights."""
    device = images.device
    prob_sum = torch.zeros(images.shape[0], 1, target_size, target_size, device=device)
    count = 0
    for scale in BEST_SCALES:
        if scale == 1.0:
            scaled = images
        else:
            sz = round(target_size * scale / 32) * 32
            scaled = F.interpolate(images, size=(sz, sz), mode="bilinear", align_corners=False)
        for aug, deaug in BEST_AUG_POOL:
            out = model(aug(scaled))
            logits = out[0] if isinstance(out, list) else out
            if scale != 1.0:
                logits = F.interpolate(logits, size=(target_size, target_size), mode="bilinear", align_corners=False)
            prob_sum += deaug(torch.sigmoid(logits))
            count += 1
    return prob_sum / count


# Simple 4-flip for validation speed (no multiscale)
_SIMPLE_TTA = [
    (lambda x: x,   lambda x: x),
    (_hflip,         _hflip),
    (_vflip,         _vflip),
    (_hvflip,        _hvflip),
]

def simple_ensemble_predict(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    prob_sum = None
    for aug, deaug in _SIMPLE_TTA:
        out = model(aug(images))
        p = deaug(torch.sigmoid(out[0] if isinstance(out, list) else out))
        prob_sum = p if prob_sum is None else prob_sum + p
    return prob_sum / len(_SIMPLE_TTA)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    data_root: str
    output_root: str
    run_name: str
    pretrained_from: str
    resume_from: str
    network: str
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    min_lr: float
    early_stopping: int
    num_workers: int
    seed: int
    fold: int
    size_rates: List[float]
    normalize_ds: bool
    sweep_only: bool
    threshold: float
    mask_threshold: int
    max_train_steps: int
    max_val_steps: int
    max_test_samples: int
    disable_cosine: bool
    ds_weights: List[float]
    ablate: str
    freeze: str
    no_pretrain: bool
    noise_aug: float = 0.0          # >0: random speckle/blur/contrast augmentation during training
    train_split: str = "train"      # "train" (fold) or "trainval" (all trainval images)
    val_split: str = "val"          # "val" (fold) or "test" (max-stretch: select on test)


def parse_args():
    root_dir = Path(__file__).resolve().parent
    default_data = root_dir / "data" / "TN3K"          # override with --data-root
    default_out = root_dir / "results"

    p = argparse.ArgumentParser(description="BiSF-Net training (E0 honest protocol, 6-head deep supervision).")
    p.add_argument("--data-root", type=Path, default=default_data)
    p.add_argument("--output-root", type=Path, default=default_out)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--pretrained-from", type=str, default="")  # E0 honest comparison: no warm-start
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--network", type=str, default="MK_UNet_Hybrid_DS_PraNet")
    p.add_argument("--fold", type=int, default=0, help="honest trainval fold for train/val (default 0)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--early-stopping", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--mask-threshold", type=int, default=MASK_THRESHOLD)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--max-val-steps", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--disable-cosine", action="store_true")
    p.add_argument("--ds-weights", type=str, default="1.0,1.0,1.0,1.0")
    p.add_argument("--normalize-ds", action=argparse.BooleanOptionalAction, default=False,
                   help="normalize DS loss by sum of weights (loss=sum(w*L)/sum(w)) for effective-LR parity across weight configs")
    p.add_argument("--sweep-only", action="store_true",
                   help="skip held-out TEST evaluation (train+checkpoint+val only) — for the DS-weight search to avoid test peeking")
    p.add_argument("--size-rates", type=str, default="0.75,1.0,1.25",
                   help="multiscale train + TTA scales; use '1.0' for fixed-native models (Swin/TransUNet/VM-UNet)")
    p.add_argument("--ablate", type=str, default="",
                   help="ablation variant for EMCADPlus / E0_PVT / E0_PVT_FullDS (e.g. noFusion, noSCSE, noRA, "
                        "noAgg, bilinear, BN, emcad / noAttn, noDS). Empty = full model.")
    p.add_argument("--freeze", type=str, default="none", choices=["none", "encoder"],
                   help="few-shot fine-tuning: 'encoder' freezes the backbone (train decoder + heads only).")
    p.add_argument("--no-pretrain", action="store_true",
                   help="random-init the PVTv2 encoder (no ImageNet pretrained weights) — for PVT-encoder models.")
    p.add_argument("--noise-aug", type=float, default=0.0,
                   help="max speckle std for random train-time noise augmentation (0 = off; also adds occasional blur/contrast)")
    p.add_argument("--train-split", type=str, default="train", choices=["train", "trainval"],
                   help="training split: 'train' (fold) or 'trainval' (all trainval images)")
    p.add_argument("--val-split", type=str, default="val", choices=["val", "test"],
                   help="validation/model-selection split: 'val' (fold) or 'test' (max-stretch)")
    return p.parse_args()


def build_run_config(args):
    pretrained = str(Path(args.pretrained_from).resolve()) if args.pretrained_from else ""
    resume = str(args.resume_from.resolve()) if args.resume_from else ""
    output_root = args.output_root.resolve()
    run_name = args.run_name

    if args.resume_from:
        rp = args.resume_from.resolve()
        if not rp.is_file():
            raise FileNotFoundError(f"Resume not found: {rp}")
        output_root = rp.parent.parent.parent
        if not run_name:
            run_name = rp.parent.parent.name

    if not run_name:
        run_name = datetime.now().strftime(
            f"ds224_pranet_lr{args.learning_rate}_bs{args.batch_size}_seed{args.seed}_%Y%m%d_%H%M%S"
        )

    return RunConfig(
        data_root=str(resolve_tn3k_root(args.data_root)),
        output_root=str(output_root), run_name=run_name,
        pretrained_from=pretrained, resume_from=resume,
        network=args.network, image_size=args.image_size,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        min_lr=args.min_lr, early_stopping=args.early_stopping,
        num_workers=args.num_workers, seed=args.seed, fold=args.fold,
        size_rates=[float(s) for s in args.size_rates.split(",")],
        threshold=args.threshold, mask_threshold=args.mask_threshold,
        max_train_steps=args.max_train_steps, max_val_steps=args.max_val_steps,
        max_test_samples=args.max_test_samples, disable_cosine=args.disable_cosine,
        ds_weights=[float(w) for w in args.ds_weights.split(",")],
        normalize_ds=args.normalize_ds, sweep_only=args.sweep_only,
        ablate=args.ablate, freeze=args.freeze, no_pretrain=args.no_pretrain,
        noise_aug=args.noise_aug, train_split=args.train_split, val_split=args.val_split,
    )


# ---------------------------------------------------------------------------
# Data: trainval (2879) for train, test (614) for val
# ---------------------------------------------------------------------------

def build_dataloaders(config: RunConfig):
    # HONEST E0 protocol: train = fold "train" (augmented), val = fold "val" (no aug,
    # used for model selection + threshold tuning), test = 614 held-out (scored once).
    gen = torch.Generator(); gen.manual_seed(config.seed)
    common = dict(num_workers=config.num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)
    train_ds = TN3KDataset(config.data_root, config.train_split, config.image_size, fold=config.fold, augment=True,
                           mask_threshold=config.mask_threshold, noise_aug=config.noise_aug)
    val_ds = TN3KDataset(config.data_root, config.val_split, config.image_size, fold=config.fold, augment=False, mask_threshold=config.mask_threshold)
    test_ds = TN3KDataset(config.data_root, "test", config.image_size, augment=False, mask_threshold=config.mask_threshold)
    return {
        "train": DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, generator=gen, **common),
        "val": DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, **common),
        "test": DataLoader(test_ds, batch_size=1, shuffle=False, **common),
    }


def build_model(config: RunConfig, device):
    """E0 comparison model registry. Only the architecture differs across models."""
    net = config.network
    root = Path(__file__).resolve().parent
    if net in ("MK_UNet_Hybrid_DS_PraNet", "MK_UNet_Hybrid_DS", "E0"):
        from mkunet_hybrid_ds_pranet import MK_UNet_Hybrid_DS_PraNet
        return MK_UNet_Hybrid_DS_PraNet(num_classes=1, in_channels=3).to(device)
    if net in ("E0_FullDS", "MK_UNet_Hybrid_DS_PraNet_FullDS"):
        from mkunet_fullds import MK_UNet_Hybrid_DS_PraNet_FullDS
        return MK_UNet_Hybrid_DS_PraNet_FullDS(num_classes=1, in_channels=3).to(device)
    if net in ("E0_RA4", "MK_UNet_RA4"):
        from mkunet_ra4 import MK_UNet_RA4
        return MK_UNet_RA4(num_classes=1, in_channels=3).to(device)
    if net in ("E0_RA6", "MK_UNet_RA6"):
        from mkunet_ra6 import MK_UNet_RA6
        return MK_UNet_RA6(num_classes=1, in_channels=3).to(device)
    if net == "FusionUNet":
        funet_root = str(root.parent / "FusionU-Net")
        if funet_root not in sys.path:
            sys.path.append(funet_root)
        from funet.FusionUNet import FusionUNet
        m = FusionUNet(in_channels=3, n_cls=1, base_channels=32).to(device)
        m.last_activation = torch.nn.Identity()   # CRITICAL: emit raw logits (disable built-in sigmoid)
        return m
    if net in ("MK_UNet", "MK_UNet_S", "MK_UNet_T"):
        from mkunet_network import MK_UNet, MK_UNet_S, MK_UNet_T
        cls = {"MK_UNet": MK_UNet, "MK_UNet_S": MK_UNet_S, "MK_UNet_T": MK_UNet_T}[net]
        return cls(num_classes=1, in_channels=3).to(device)
    if net == "PraNet":
        from pranet_e0 import PraNet_E0
        return PraNet_E0(num_classes=1, in_channels=3).to(device)
    if net == "SwinUNet":
        from swinunet_e0 import SwinUnet_E0
        return SwinUnet_E0(num_classes=1, in_channels=3, img_size=config.image_size).to(device)
    if net == "TransUNet":
        from transunet_e0 import TransUNet_E0
        return TransUNet_E0(num_classes=1, in_channels=3, img_size=config.image_size).to(device)
    if net == "MambaUNet":
        from mambaunet_e0 import MambaUnet_E0
        return MambaUnet_E0(num_classes=1, in_channels=3, img_size=config.image_size).to(device)
    if net == "VMUNet":
        from vmunet_e0 import VMUNet_E0
        return VMUNet_E0(num_classes=1, in_channels=3, img_size=config.image_size).to(device)
    if net == "nnUNet":
        from nnunet_e0 import nnUNet_E0
        return nnUNet_E0(num_classes=1, in_channels=3, img_size=config.image_size).to(device)
    if net == "nnWNet":
        from nnwnet_e0 import nnWNet_E0
        return nnWNet_E0(num_classes=1, in_channels=3, img_size=config.image_size).to(device)
    if net == "TransAttUnet":
        from transattunet_e0 import TransAttUnet_E0
        return TransAttUnet_E0(num_classes=1, in_channels=3, img_size=config.image_size).to(device)
    if net == "MDPNet":
        from mdpnet_e0 import MDPNet_E0
        return MDPNet_E0(num_classes=1, in_channels=3, img_size=config.image_size).to(device)
    if net == "YoloSeg":
        from yoloseg_e0 import YoloSeg_E0
        return YoloSeg_E0(num_classes=1, in_channels=3, img_size=config.image_size).to(device)
    if net == "CFCM":
        from cfcm_e0 import CFCM_E0
        return CFCM_E0(num_classes=1, in_channels=3, img_size=config.image_size).to(device)
    if net == "MSDUNet":
        from msdunet_e0 import MSDUNet_E0
        return MSDUNet_E0(num_classes=1, in_channels=3, img_size=config.image_size).to(device)
    if net == "EMCAD":
        from emcad_e0 import EMCAD_E0
        return EMCAD_E0(num_classes=1, in_channels=3, img_size=config.image_size, no_pretrain=config.no_pretrain).to(device)
    if net == "EMCAD_B0":
        from emcad_e0 import EMCAD_E0
        return EMCAD_E0(num_classes=1, in_channels=3, img_size=config.image_size,
                        encoder="pvt_v2_b0", no_pretrain=config.no_pretrain).to(device)
    if net == "EMCADPlus":
        from emcadplus_e0 import EMCADPlus_E0
        return EMCADPlus_E0(num_classes=1, in_channels=3, img_size=config.image_size, ablate=config.ablate,
                            no_pretrain=config.no_pretrain).to(device)
    if net == "EMCADPlus_B0":
        from emcadplus_e0 import EMCADPlus_E0
        return EMCADPlus_E0(num_classes=1, in_channels=3, img_size=config.image_size, ablate=config.ablate,
                            encoder="pvt_v2_b0", no_pretrain=config.no_pretrain).to(device)
    if net == "EMCADPlus_CNX":
        from emcadplus_e0 import EMCADPlus_E0
        return EMCADPlus_E0(num_classes=1, in_channels=3, img_size=config.image_size, ablate=config.ablate,
                            encoder="convnext_tiny", no_pretrain=config.no_pretrain).to(device)
    if net == "EMCADPlus_RN50":
        from emcadplus_e0 import EMCADPlus_E0
        return EMCADPlus_E0(num_classes=1, in_channels=3, img_size=config.image_size, ablate=config.ablate,
                            encoder="resnet50", no_pretrain=config.no_pretrain).to(device)
    if net == "BAANPlus":
        from baanplus_e0 import BAANPlus_E0
        return BAANPlus_E0(num_classes=1, in_channels=3, img_size=config.image_size, ablate=config.ablate,
                           no_pretrain=config.no_pretrain).to(device)
    if net in ("E0_PVT", "MK_UNet_PVT_DS_PraNet"):
        from mkunet_pvt_e0 import MK_UNet_PVT_DS_PraNet
        return MK_UNet_PVT_DS_PraNet(num_classes=1, in_channels=3, ablate=config.ablate,
                                     no_pretrain=config.no_pretrain).to(device)
    if net == "E0_PVT_B0":
        from mkunet_pvt_e0 import MK_UNet_PVT_DS_PraNet
        return MK_UNet_PVT_DS_PraNet(num_classes=1, in_channels=3, ablate=config.ablate,
                                     variant="b0", no_pretrain=config.no_pretrain).to(device)
    if net == "E0_PVT_CNX":
        from mkunet_pvt_e0 import MK_UNet_PVT_DS_PraNet
        return MK_UNet_PVT_DS_PraNet(num_classes=1, in_channels=3, ablate=config.ablate,
                                     variant="convnext_tiny", no_pretrain=config.no_pretrain).to(device)
    if net == "E0_PVT_RN50":
        from mkunet_pvt_e0 import MK_UNet_PVT_DS_PraNet
        return MK_UNet_PVT_DS_PraNet(num_classes=1, in_channels=3, ablate=config.ablate,
                                     variant="resnet50", no_pretrain=config.no_pretrain).to(device)
    if net in ("E0_PVT_FullDS", "MK_UNet_PVT_FullDS"):
        from mkunet_pvt_fullds import MK_UNet_PVT_FullDS
        return MK_UNet_PVT_FullDS(num_classes=1, in_channels=3, ablate=config.ablate,
                                  no_pretrain=config.no_pretrain).to(device)
    if net == "E0_PVT_FullDS_B0":
        from mkunet_pvt_fullds import MK_UNet_PVT_FullDS
        return MK_UNet_PVT_FullDS(num_classes=1, in_channels=3, ablate=config.ablate,
                                  variant="b0", no_pretrain=config.no_pretrain).to(device)
    if net == "E0_PVT_FullDS_CNX":
        from mkunet_pvt_fullds import MK_UNet_PVT_FullDS
        return MK_UNet_PVT_FullDS(num_classes=1, in_channels=3, ablate=config.ablate,
                                  variant="convnext_tiny", no_pretrain=config.no_pretrain).to(device)
    if net == "E0_PVT_FullDS_RN50":
        from mkunet_pvt_fullds import MK_UNet_PVT_FullDS
        return MK_UNet_PVT_FullDS(num_classes=1, in_channels=3, ablate=config.ablate,
                                  variant="resnet50", no_pretrain=config.no_pretrain).to(device)
    raise ValueError(f"Unknown --network for E0 honest comparison: {net}")


# ---------------------------------------------------------------------------
# Epoch pass with PraNet-style multi-scale training
# ---------------------------------------------------------------------------

def _resize_batch_for_rate(images, masks, base_size: int, rate: float):
    target_size = int(round(base_size * rate / 32) * 32)
    if target_size == base_size:
        return images, masks
    return (
        F.interpolate(images, size=(target_size, target_size), mode="bilinear", align_corners=False),
        F.interpolate(masks, size=(target_size, target_size), mode="nearest"),
    )


def epoch_pass(model, loader, device, threshold, ds_weights, optimizer=None, max_steps=0, epoch=0, total_epochs=0, normalize_ds=False):
    is_training = optimizer is not None
    model.train(is_training)
    n_branches = len(BRANCH_NAMES)
    loss_sum = 0.0; loss_count = 0
    branch_loss_sums = [0.0] * n_branches
    branch_dice_sums = [0.0] * n_branches
    branch_count = 0
    accumulator = MetricAccumulator()
    stage = "Train" if is_training else "Val"
    total_batches = min(len(loader), max_steps) if max_steps else len(loader)
    progress = tqdm(total=total_batches, desc=f"Epoch {epoch}/{total_epochs} {stage}", leave=False, dynamic_ncols=True)

    try:
        for step, batch in enumerate(loader, 1):
            if max_steps and step > max_steps: break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            bs = images.shape[0]

            if is_training:
                primary = None
                primary_loss = None
                outputs = None
                for rate in PRANET_SIZE_RATES:
                    scaled_images, scaled_masks = _resize_batch_for_rate(images, masks, images.shape[-1], rate)
                    with torch.set_grad_enabled(True):
                        outputs = model(scaled_images)
                        if isinstance(outputs, list) and len(outputs) > 1:
                            ws = ds_weights[:len(outputs)]
                            loss = sum(w * structure_loss(o, scaled_masks) for w, o in zip(ws, outputs))
                            if normalize_ds:
                                loss = loss / sum(ws)
                        else:
                            logits = outputs[0] if isinstance(outputs, list) else outputs
                            loss = structure_loss(logits, scaled_masks)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()

                    if rate == 1.0:
                        primary_loss = loss.item()
                        primary = (outputs[0] if isinstance(outputs, list) else outputs).detach()
                        if isinstance(outputs, list) and len(outputs) > 1:
                            with torch.no_grad():
                                targets = (masks >= 0.5).float()
                                dims = (1, 2, 3)
                                for bi in range(min(n_branches, len(outputs))):
                                    branch_out = outputs[bi].detach()
                                    branch_loss_sums[bi] += structure_loss(branch_out, masks).item() * bs
                                    probs = torch.sigmoid(branch_out)
                                    preds = (probs >= threshold).float()
                                    tp = (preds * targets).sum(dim=dims)
                                    fp = (preds * (1 - targets)).sum(dim=dims)
                                    fn = ((1 - preds) * targets).sum(dim=dims)
                                    branch_dice_sums[bi] += ((2 * tp + EPS) / (2 * tp + fp + fn + EPS)).mean().item() * bs
                            branch_count += bs
                if primary is None or primary_loss is None:
                    raise RuntimeError("PraNet-style training pass did not produce a primary output.")
            else:
                with torch.set_grad_enabled(False):
                    outputs = model(images)
                    logits = outputs[0] if isinstance(outputs, list) else outputs
                    loss = structure_loss(logits, masks)

                if isinstance(outputs, list) and len(outputs) > 1:
                    with torch.no_grad():
                        targets = (masks >= 0.5).float()
                        dims = (1, 2, 3)
                        for bi in range(min(n_branches, len(outputs))):
                            branch_out = outputs[bi].detach()
                            branch_loss_sums[bi] += structure_loss(branch_out, masks).item() * bs
                            probs = torch.sigmoid(branch_out)
                            preds = (probs >= threshold).float()
                            tp = (preds * targets).sum(dim=dims)
                            fp = (preds * (1 - targets)).sum(dim=dims)
                            fn = ((1 - preds) * targets).sum(dim=dims)
                            branch_dice_sums[bi] += ((2 * tp + EPS) / (2 * tp + fp + fn + EPS)).mean().item() * bs
                    branch_count += bs
                primary = logits.detach()

            accumulator.update(batch_metrics(torch.sigmoid(primary), masks.detach(), threshold))
            loss_sum += (primary_loss if is_training else loss.item()) * bs
            loss_count += bs

            running = accumulator.averages()
            pf = dict(loss=f"{loss_sum/max(loss_count,1):.4f}", dice=f"{running.get('dice',0):.4f}", iou=f"{running.get('iou',0):.4f}")
            if is_training: pf["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
            progress.set_postfix(**pf); progress.update(1)
    finally:
        progress.close()

    denom = max(loss_count, 1); bdenom = max(branch_count, 1)
    results = accumulator.averages()
    results["loss"] = loss_sum / denom
    for bi, name in enumerate(BRANCH_NAMES):
        results[f"{name}_loss"] = branch_loss_sums[bi] / bdenom
        results[f"{name}_dice"] = branch_dice_sums[bi] / bdenom
    return results


# ---------------------------------------------------------------------------
# Threshold tuning (best ensemble)
# ---------------------------------------------------------------------------

def tune_threshold(model, loader, device, image_size, logger, max_batches=0):
    model.eval()
    all_p, all_m = [], []
    with torch.inference_mode():
        for batch_idx, b in enumerate(tqdm(loader, desc="Threshold tuning (best ensemble)", leave=False, dynamic_ncols=True), start=1):
            if max_batches and batch_idx > max_batches:
                break
            imgs = b["image"].to(device, non_blocking=True)
            all_p.append(best_ensemble_predict(model, imgs, image_size).cpu())
            all_m.append(b["mask"])
    if not all_p:
        logger.warning("Threshold tuning skipped because no validation batches were processed; using default threshold 0.50.")
        return 0.5
    pt, mt = torch.cat(all_p), (torch.cat(all_m) >= 0.5).float()
    best_t, best_d = 0.5, 0.0; dims = (1, 2, 3)
    for ti in range(1, 100):
        t = ti / 100.0; pr = (pt >= t).float()
        tp = (pr * mt).sum(dim=dims); fp = (pr * (1 - mt)).sum(dim=dims); fn = ((1 - pr) * mt).sum(dim=dims)
        d = ((2*tp+EPS)/(2*tp+fp+fn+EPS)).mean().item()
        if d > best_d: best_d, best_t = d, t
    logger.info("Tuned threshold: %.2f (best-ensemble Dice: %.4f)", best_t, best_d)
    return best_t


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(model, loader, device, threshold, mask_threshold, output_dir, image_size, max_samples=0, use_best_ensemble=False):
    model.eval()
    for d in ("predicted_masks", "ground_truth_masks", "probability_maps", "visualizations"):
        (output_dir / d).mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for si, batch in enumerate(tqdm(loader, desc="Test eval", leave=False, dynamic_ncols=True), 1):
            if max_samples and si > max_samples: break
            imgs = batch["image"].to(device)
            if use_best_ensemble:
                probs_np = best_ensemble_predict(model, imgs, image_size).cpu().numpy()
            else:
                out = model(imgs)
                logits = out[0] if isinstance(out, list) else out   # bare-tensor models (Swin/Fusion/TransUNet/Mamba) vs list-head models
                probs_np = torch.sigmoid(logits).detach().cpu().numpy()
            for bi in range(imgs.shape[0]):
                sid = batch["sample_id"][bi]
                ip, mp = Path(batch["image_path"][bi]), Path(batch["mask_path"][bi])
                orig = np.asarray(Image.open(ip).convert("RGB")); oh, ow = orig.shape[:2]
                gt = (np.asarray(Image.open(mp).convert("L"), dtype=np.uint8) > mask_threshold).astype(np.uint8)
                pm = (probs_np[bi, 0] * 255).astype(np.uint8)
                resized = np.asarray(Image.fromarray(pm).resize((ow, oh), resample=Image.Resampling.BILINEAR), dtype=np.uint8)
                pred = (resized >= int(threshold * 255)).astype(np.uint8)
                m = numpy_metrics(pred, gt)
                pp = output_dir/"predicted_masks"/f"{sid}.png"; gp = output_dir/"ground_truth_masks"/f"{sid}.png"
                prp = output_dir/"probability_maps"/f"{sid}.png"; vp = output_dir/"visualizations"/f"{sid}.png"
                Image.fromarray(pred*255).save(pp); Image.fromarray(gt*255).save(gp)
                Image.fromarray(resized).save(prp); build_visualization(orig, gt, pred).save(vp)
                records.append({"sample_id": sid, "image_path": str(ip), "predicted_mask_path": str(pp),
                                "ground_truth_mask_path": str(gp), "probability_map_path": str(prp),
                                "visualization_path": str(vp), **m})
    mn = ["dice","iou","precision","recall","specificity","accuracy"]
    summary = {"num_samples": len(records), "threshold": threshold, "ensemble": use_best_ensemble,
               "mean": {k: float(np.mean([r[k] for r in records])) if records else 0 for k in mn},
               "std": {k: float(np.std([r[k] for r in records])) if records else 0 for k in mn}}
    if records:
        with (output_dir/"per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)
    with (output_dir/"summary.json").open("w", encoding="utf-8") as f: json.dump(summary, f, indent=2)
    with (output_dir/"summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stat"]+mn); w.writeheader()
        w.writerow({"stat":"mean",**summary["mean"]}); w.writerow({"stat":"std",**summary["std"]})
    return {"records": records, "summary": summary}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def train_and_evaluate(config: RunConfig) -> Path:
    global PRANET_SIZE_RATES, BEST_SCALES
    PRANET_SIZE_RATES = list(config.size_rates)   # multiscale training rates
    BEST_SCALES = list(config.size_rates)         # TTA scales (match training)
    run_dir = Path(config.output_root) / config.run_name
    ckpt_dir = run_dir / "checkpoints"
    test_dir = run_dir / "test_results"
    test_ens_dir = run_dir / "test_results_best_ensemble"
    for d in (ckpt_dir, test_dir, test_ens_dir): d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | Image size: %d | Batch size: %d", device, config.image_size, config.batch_size)
    logger.info("Pretrained: %s", config.pretrained_from)

    seed_everything(config.seed)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    dataloaders = build_dataloaders(config)
    logger.info("HONEST fold%d: Train %d (train) | Val %d (val) | Test %d (held-out, scored once) | Network %s",
                config.fold, len(dataloaders["train"].dataset), len(dataloaders["val"].dataset),
                len(dataloaders["test"].dataset), config.network)

    model = build_model(config, device)

    # Warm-start from pretrained (strict=False: ds_out3 dropped, sg_out new)
    if config.pretrained_from:
        ckpt = torch.load(config.pretrained_from, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        logger.info("Loaded pretrained weights (epoch %d, val dice %.4f)", ckpt.get("epoch", -1), ckpt.get("best_val_dice", -1))
        if missing:
            logger.info("Missing keys (randomly initialized): %s", missing)
        if unexpected:
            logger.info("Unexpected keys (skipped): %s", unexpected)

    # Few-shot fine-tuning: optionally freeze the pretrained encoder (train decoder + heads only).
    # Matches on encoder-module name keywords across families. (Transformer encoders use LayerNorm =>
    # clean freeze; CNN-encoder baselines may still drift BN running stats slightly — acceptable, the
    # freeze x lr grid selects whichever variant wins on val.)
    if getattr(config, "freeze", "none") == "encoder":
        enc_kw = ("encoder", "backbone", "patch_embed", "pvt", ".inc.", "down1", "down2",
                  "down3", "down4", ".layers.", ".blocks.")
        nfrozen = 0
        for name, prm in model.named_parameters():
            if any(k in f".{name.lower()}." for k in enc_kw):
                prm.requires_grad_(False); nfrozen += 1
        ntrain = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info("FREEZE=encoder: froze %d encoder params; %d param-tensors remain trainable", nfrozen, ntrain)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = None if config.disable_cosine else CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)

    best_ckpt = ckpt_dir / "best_model.pt"
    hist_csv = run_dir / "history.csv"

    start_epoch, best_val_dice, no_improve = 1, -1.0, 0
    history: List[Dict] = []
    if config.resume_from:
        rckpt = torch.load(config.resume_from, map_location=device)
        model.load_state_dict(rckpt["model_state"])
        if rckpt.get("optimizer_state"): optimizer.load_state_dict(rckpt["optimizer_state"])
        if scheduler and rckpt.get("scheduler_state"): scheduler.load_state_dict(rckpt["scheduler_state"])
        start_epoch = int(rckpt["epoch"]) + 1
        best_val_dice = float(rckpt.get("best_val_dice", -1.0))
        no_improve = int(rckpt.get("epochs_without_improvement", 0))
        history = load_history_csv(hist_csv)
        history = [r for r in history if int(r["epoch"]) <= rckpt["epoch"]]
        logger.info("Resumed from epoch %d", rckpt["epoch"])

    for epoch in range(start_epoch, config.epochs + 1):
        logger.info("Starting epoch %03d/%03d", epoch, config.epochs)
        train_m = epoch_pass(model, dataloaders["train"], device, config.threshold, config.ds_weights, normalize_ds=config.normalize_ds,
                             optimizer=optimizer, max_steps=config.max_train_steps, epoch=epoch, total_epochs=config.epochs)
        val_m = epoch_pass(model, dataloaders["val"], device, config.threshold, config.ds_weights,
                           optimizer=None, max_steps=config.max_val_steps, epoch=epoch, total_epochs=config.epochs)
        if scheduler: scheduler.step()

        logger.info("Epoch %03d | t_loss %.4f | t_dice %.4f | v_loss %.4f | v_dice %.4f | v_iou %.4f",
                     epoch, train_m["loss"], train_m["dice"], val_m["loss"], val_m["dice"], val_m["iou"])
        for bn in BRANCH_NAMES:
            logger.info("  %-4s | t_loss %.4f  t_dice %.4f | v_loss %.4f  v_dice %.4f",
                         bn, train_m.get(f"{bn}_loss",0), train_m.get(f"{bn}_dice",0),
                         val_m.get(f"{bn}_loss",0), val_m.get(f"{bn}_dice",0))

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        for k, v in train_m.items(): row[f"train_{k}"] = v
        for k, v in val_m.items(): row[f"val_{k}"] = v
        history.append(row); write_history_csv(history, hist_csv)

        if val_m["dice"] > best_val_dice:
            best_val_dice = val_m["dice"]; no_improve = 0
            save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch, best_val_dice, no_improve, config)
            logger.info("New best val dice: %.4f", best_val_dice)
        else:
            no_improve += 1
            logger.info("No improvement %d/%d", no_improve, config.early_stopping)

        save_checkpoint(ckpt_dir/"last_model.pt", model, optimizer, scheduler, epoch, best_val_dice, no_improve, config)
        if no_improve >= config.early_stopping:
            logger.info("Early stopping."); break

    # --- Load best and run final evaluation ---
    if best_ckpt.is_file():
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state"])

    if config.sweep_only:
        # DS-weight search: NO test evaluation (avoid test peeking). Record val-selection only.
        with (run_dir/"sweep_val.json").open("w", encoding="utf-8") as f:
            json.dump({"best_val_dice": best_val_dice, "ds_weights": config.ds_weights,
                       "normalize_ds": config.normalize_ds, "run_name": config.run_name}, f, indent=2)
        logger.info("SWEEP-ONLY: best_val_dice %.4f (no test eval). run=%s", best_val_dice, config.run_name)
        return run_dir

    # Standard test (no TTA)
    logger.info("Standard test (t=%.2f) ...", config.threshold)
    std = evaluate_test_set(model, dataloaders["test"], device, config.threshold, config.mask_threshold, test_dir, config.image_size)
    logger.info("Standard — Dice: %.4f  IoU: %.4f", std["summary"]["mean"]["dice"], std["summary"]["mean"]["iou"])

    # Best ensemble + fine threshold
    logger.info("Tuning threshold with best ensemble (8 flip+rot × 3 scales) ...")
    tuned = tune_threshold(model, dataloaders["val"], device, config.image_size, logger, max_batches=config.max_val_steps)

    logger.info("Best ensemble test (t=%.2f) ...", tuned)
    ens = evaluate_test_set(model, dataloaders["test"], device, tuned, config.mask_threshold, test_ens_dir, config.image_size, use_best_ensemble=True)
    logger.info("Best ensemble — Dice: %.4f  IoU: %.4f", ens["summary"]["mean"]["dice"], ens["summary"]["mean"]["iou"])

    with (run_dir/"test_comparison.json").open("w", encoding="utf-8") as f:
        json.dump({"standard": std["summary"], "best_ensemble": ens["summary"], "tuned_threshold": tuned,
                    "ensemble_config": "8flip_rot_ms3 (8 augs × 3 scales [0.75, 1.0, 1.25])"}, f, indent=2)
    logger.info("Done. Outputs: %s", run_dir)
    return run_dir


def main():
    config = build_run_config(parse_args())
    train_and_evaluate(config)

if __name__ == "__main__":
    main()
