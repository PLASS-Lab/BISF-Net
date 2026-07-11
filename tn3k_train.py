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
from typing import Dict, List

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.backends import cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mkunet_fusion_network import MK_UNet_FFM
from mkunet_hybrid_network import MK_UNet_Hybrid
from mkunet_network import MK_UNet, MK_UNet_S, MK_UNet_T
from tn3k_dataset import MASK_THRESHOLD, TN3KSegmentationDataset, resolve_tn3k_root, summarize_tn3k


EPS = 1e-7


@dataclass
class RunConfig:
    data_root: str
    output_root: str
    run_name: str
    resume_from: str
    network: str
    fold: int
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    min_lr: float
    early_stopping: int
    num_workers: int
    seed: int
    threshold: float
    mask_threshold: int
    max_train_steps: int
    max_val_steps: int
    max_test_samples: int
    disable_cosine: bool


NETWORK_BUILDERS = {
    "MK_UNet_T": MK_UNet_T,
    "MK_UNet_S": MK_UNet_S,
    "MK_UNet": MK_UNet,
    "MK_UNet_M": lambda **kwargs: MK_UNet(channels=[32, 64, 128, 192, 320], **kwargs),
    "MK_UNet_L": lambda **kwargs: MK_UNet(channels=[64, 128, 256, 384, 512], **kwargs),
    "MK_UNet_FFM": MK_UNet_FFM,
    "MK_UNet_Hybrid": MK_UNet_Hybrid,
}


class MetricAccumulator:
    def __init__(self) -> None:
        self.metric_sums = defaultdict(float)
        self.count = 0

    def update(self, metrics: Dict[str, torch.Tensor]) -> None:
        batch_count = int(next(iter(metrics.values())).numel())
        for name, values in metrics.items():
            self.metric_sums[name] += float(values.detach().sum().item())
        self.count += batch_count

    def averages(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        return {name: total / self.count for name, total in self.metric_sums.items()}


def structure_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weighted_bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    weighted_bce = (weights * weighted_bce).sum(dim=(2, 3)) / weights.sum(dim=(2, 3))

    probabilities = torch.sigmoid(logits)
    intersection = ((probabilities * mask) * weights).sum(dim=(2, 3))
    union = ((probabilities + mask) * weights).sum(dim=(2, 3))
    weighted_iou = 1 - (intersection + 1) / (union - intersection + 1)

    return (weighted_bce + weighted_iou).mean()


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    default_data_root = root_dir.parent / "Datasets" / "TN3K" / "raw"
    default_output_root = root_dir / "results" / "TN3K" / "MK_UNet"

    parser = argparse.ArgumentParser(
        description="Train MK-UNet on TN3K and run test-time evaluation automatically."
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--network", type=str, default="MK_UNet", choices=list(NETWORK_BUILDERS.keys()))
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=int, default=MASK_THRESHOLD)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--disable-cosine", action="store_true")
    return parser.parse_args()


def build_run_config(args: argparse.Namespace) -> RunConfig:
    resume_from = str(args.resume_from.resolve()) if args.resume_from else ""
    output_root = args.output_root.resolve()
    run_name = args.run_name

    if args.resume_from:
        resume_path = args.resume_from.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        inferred_run_dir = resume_path.parent.parent
        output_root = inferred_run_dir.parent
        if not run_name:
            run_name = inferred_run_dir.name

    if not run_name:
        run_name = datetime.now().strftime(
            f"tn3k_{args.network}_fold{args.fold}_seed{args.seed}_%Y%m%d_%H%M%S"
        )

    return RunConfig(
        data_root=str(resolve_tn3k_root(args.data_root)),
        output_root=str(output_root),
        run_name=run_name,
        resume_from=resume_from,
        network=args.network,
        fold=args.fold,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        min_lr=args.min_lr,
        early_stopping=args.early_stopping,
        num_workers=args.num_workers,
        seed=args.seed,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        max_test_samples=args.max_test_samples,
        disable_cosine=args.disable_cosine,
    )


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("mkunet_tn3k_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def build_model(config: RunConfig, device: torch.device) -> nn.Module:
    model = NETWORK_BUILDERS[config.network](num_classes=1, in_channels=3)
    return model.to(device)


def build_dataloaders(config: RunConfig) -> Dict[str, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_dataset = TN3KSegmentationDataset(
        data_root=config.data_root,
        split="train",
        image_size=config.image_size,
        fold=config.fold,
        augment=True,
        mask_threshold=config.mask_threshold,
    )
    val_dataset = TN3KSegmentationDataset(
        data_root=config.data_root,
        split="val",
        image_size=config.image_size,
        fold=config.fold,
        augment=False,
        mask_threshold=config.mask_threshold,
    )
    test_dataset = TN3KSegmentationDataset(
        data_root=config.data_root,
        split="test",
        image_size=config.image_size,
        fold=config.fold,
        augment=False,
        mask_threshold=config.mask_threshold,
    )

    common_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            **common_kwargs,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            **common_kwargs,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            **common_kwargs,
        ),
    }


def batch_metrics(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float) -> Dict[str, torch.Tensor]:
    predictions = (probabilities >= threshold).float()
    targets = (targets >= 0.5).float()
    dims = (1, 2, 3)
    tp = (predictions * targets).sum(dim=dims)
    fp = (predictions * (1 - targets)).sum(dim=dims)
    fn = ((1 - predictions) * targets).sum(dim=dims)
    tn = ((1 - predictions) * (1 - targets)).sum(dim=dims)
    return {
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "iou": (tp + EPS) / (tp + fp + fn + EPS),
        "precision": (tp + EPS) / (tp + fp + EPS),
        "recall": (tp + EPS) / (tp + fn + EPS),
        "specificity": (tn + EPS) / (tn + fp + EPS),
        "accuracy": (tp + tn + EPS) / (tp + tn + fp + fn + EPS),
    }


def numpy_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = prediction.astype(bool)
    gt = target.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), gt).sum())
    tn = float(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    return {
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "iou": (tp + EPS) / (tp + fp + fn + EPS),
        "precision": (tp + EPS) / (tp + fp + EPS),
        "recall": (tp + EPS) / (tp + fn + EPS),
        "specificity": (tn + EPS) / (tn + fp + EPS),
        "accuracy": (tp + tn + EPS) / (tp + tn + fp + fn + EPS),
    }


def get_primary_logits(outputs: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
    return outputs[0] if isinstance(outputs, list) else outputs


def epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int = 0,
    epoch: int = 0,
    total_epochs: int = 0,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    loss_sum = 0.0
    loss_count = 0
    accumulator = MetricAccumulator()
    stage_name = "Train" if is_training else "Val"
    total_batches = len(loader)
    if max_steps:
        total_batches = min(total_batches, max_steps)

    progress = tqdm(
        total=total_batches,
        desc=f"Epoch {epoch}/{total_epochs} {stage_name}",
        leave=False,
        dynamic_ncols=True,
    )

    try:
        for step, batch in enumerate(loader, start=1):
            if max_steps and step > max_steps:
                break

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.set_grad_enabled(is_training):
                logits = get_primary_logits(model(images))
                loss = structure_loss(logits, masks)
                if is_training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            probabilities = torch.sigmoid(logits.detach())
            metrics = batch_metrics(probabilities, masks.detach(), threshold)
            accumulator.update(metrics)
            batch_size = images.shape[0]
            loss_sum += loss.item() * batch_size
            loss_count += batch_size

            running_loss = loss_sum / max(loss_count, 1)
            running_metrics = accumulator.averages()
            progress.set_postfix(
                loss=f"{running_loss:.4f}",
                dice=f"{running_metrics.get('dice', 0.0):.4f}",
                iou=f"{running_metrics.get('iou', 0.0):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}" if is_training else "-",
            )
            progress.update(1)
    finally:
        progress.close()

    results = accumulator.averages()
    results["loss"] = loss_sum / max(loss_count, 1)
    return results


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR | None,
    epoch: int,
    best_val_dice: float,
    epochs_without_improvement: int,
    config: RunConfig,
) -> None:
    payload = {
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "epochs_without_improvement": epochs_without_improvement,
        "config": asdict(config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(payload, checkpoint_path)


def write_history_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_history_csv(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.is_file():
        return []
    rows: List[Dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: Dict[str, object] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                    continue
                try:
                    number = float(value)
                    parsed[key] = int(number) if number.is_integer() else number
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def infer_epochs_without_improvement(history_rows: List[Dict[str, object]]) -> tuple[float, int]:
    if not history_rows:
        return -1.0, 0
    best_val_dice = -1.0
    best_epoch = 0
    last_epoch = int(history_rows[-1]["epoch"])
    for row in history_rows:
        val_dice = float(row["val_dice"])
        epoch = int(row["epoch"])
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
    return best_val_dice, last_epoch - best_epoch


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.astype(np.float32).copy()
    mask_indices = mask.astype(bool)
    if np.any(mask_indices):
        result[mask_indices] = result[mask_indices] * 0.55 + np.array(color, dtype=np.float32) * 0.45
    return result.clip(0, 255).astype(np.uint8)


def build_visualization(image: np.ndarray, ground_truth: np.ndarray, prediction: np.ndarray) -> Image.Image:
    gt_overlay = overlay_mask(image, ground_truth, (0, 255, 0))
    pred_overlay = overlay_mask(image, prediction, (255, 0, 0))

    error_overlay = image.copy()
    true_positive = np.logical_and(ground_truth, prediction)
    false_positive = np.logical_and(np.logical_not(ground_truth), prediction)
    false_negative = np.logical_and(ground_truth, np.logical_not(prediction))
    error_overlay = overlay_mask(error_overlay, true_positive, (0, 255, 0))
    error_overlay = overlay_mask(error_overlay, false_positive, (255, 0, 0))
    error_overlay = overlay_mask(error_overlay, false_negative, (0, 0, 255))

    labeled_panels: List[Image.Image] = []
    for label, panel in [
        ("Original", image),
        ("GT Overlay", gt_overlay),
        ("Pred Overlay", pred_overlay),
        ("Error Map", error_overlay),
    ]:
        panel_image = Image.fromarray(panel)
        canvas = Image.new("RGB", (panel_image.width, panel_image.height + 24), "white")
        canvas.paste(panel_image, (0, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), label, fill="black", font=ImageFont.load_default())
        labeled_panels.append(canvas)

    strip = Image.new(
        "RGB",
        (sum(panel.width for panel in labeled_panels), labeled_panels[0].height),
        "white",
    )
    current_x = 0
    for panel in labeled_panels:
        strip.paste(panel, (current_x, 0))
        current_x += panel.width
    return strip


def evaluate_test_set(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    mask_threshold: int,
    output_dir: Path,
    max_samples: int = 0,
) -> Dict[str, object]:
    model.eval()
    predicted_mask_dir = output_dir / "predicted_masks"
    ground_truth_dir = output_dir / "ground_truth_masks"
    probability_dir = output_dir / "probability_maps"
    visualization_dir = output_dir / "visualizations"
    predicted_mask_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    probability_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    with torch.inference_mode():
        for sample_index, batch in enumerate(loader, start=1):
            if max_samples and sample_index > max_samples:
                break

            images = batch["image"].to(device, non_blocking=True)
            logits = get_primary_logits(model(images))
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()

            for batch_index in range(images.shape[0]):
                sample_id = batch["sample_id"][batch_index]
                image_path = Path(batch["image_path"][batch_index])
                mask_path = Path(batch["mask_path"][batch_index])

                original_image = np.asarray(Image.open(image_path).convert("RGB"))
                original_height, original_width = original_image.shape[:2]
                ground_truth = (
                    np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > mask_threshold
                ).astype(np.uint8)

                probability_map = (probabilities[batch_index, 0] * 255).astype(np.uint8)
                resized_probability = Image.fromarray(probability_map).resize(
                    (original_width, original_height), resample=Image.Resampling.BILINEAR
                )
                probability_array = np.asarray(resized_probability, dtype=np.uint8)
                prediction = (probability_array >= int(threshold * 255)).astype(np.uint8)

                metrics = numpy_metrics(prediction, ground_truth)

                predicted_mask_path = predicted_mask_dir / f"{sample_id}.png"
                ground_truth_mask_path = ground_truth_dir / f"{sample_id}.png"
                probability_path = probability_dir / f"{sample_id}.png"
                visualization_path = visualization_dir / f"{sample_id}.png"

                Image.fromarray(prediction * 255).save(predicted_mask_path)
                Image.fromarray(ground_truth * 255).save(ground_truth_mask_path)
                Image.fromarray(probability_array).save(probability_path)
                build_visualization(original_image, ground_truth, prediction).save(visualization_path)

                records.append(
                    {
                        "sample_id": sample_id,
                        "image_path": str(image_path),
                        "ground_truth_source_path": str(mask_path),
                        "predicted_mask_path": str(predicted_mask_path),
                        "ground_truth_mask_path": str(ground_truth_mask_path),
                        "probability_map_path": str(probability_path),
                        "visualization_path": str(visualization_path),
                        **metrics,
                    }
                )

    metric_names = ["dice", "iou", "precision", "recall", "specificity", "accuracy"]
    summary = {
        "num_samples": len(records),
        "mean": {
            metric: float(np.mean([record[metric] for record in records])) if records else 0.0
            for metric in metric_names
        },
        "std": {
            metric: float(np.std([record[metric] for record in records])) if records else 0.0
            for metric in metric_names
        },
    }

    metrics_csv = output_dir / "per_sample_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()) if records else ["sample_id"])
        writer.writeheader()
        if records:
            writer.writerows(records)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stat"] + metric_names)
        writer.writeheader()
        writer.writerow({"stat": "mean", **summary["mean"]})
        writer.writerow({"stat": "std", **summary["std"]})

    return {"records": records, "summary": summary}


def maybe_resume_training(
    config: RunConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR | None,
    history_csv_path: Path,
    logger: logging.Logger,
    device: torch.device,
) -> tuple[int, float, int, List[Dict[str, object]]]:
    history_rows = load_history_csv(history_csv_path)
    if not config.resume_from:
        return 1, -1.0, 0, history_rows

    checkpoint = torch.load(config.resume_from, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    checkpoint_epoch = int(checkpoint["epoch"])
    history_rows = [row for row in history_rows if int(row["epoch"]) <= checkpoint_epoch]

    if checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if hasattr(scheduler, "T_max") and scheduler.T_max != config.epochs:
            old_t_max = scheduler.T_max
            scheduler.T_max = config.epochs
            logger.info(
                "Adjusted cosine scheduler T_max from %s to %s for resumed training.",
                old_t_max,
                config.epochs,
            )

    start_epoch = checkpoint_epoch + 1
    checkpoint_best = float(checkpoint.get("best_val_dice", -1.0))
    checkpoint_counter = int(checkpoint.get("epochs_without_improvement", -1))
    inferred_best, inferred_counter = infer_epochs_without_improvement(history_rows)
    best_val_dice = checkpoint_best if checkpoint_best >= inferred_best else inferred_best
    epochs_without_improvement = checkpoint_counter if checkpoint_counter >= 0 else inferred_counter

    logger.info("Resuming training from checkpoint: %s", config.resume_from)
    logger.info(
        "Resume state | next epoch: %d | best val dice: %.4f | no-improvement counter: %d",
        start_epoch,
        best_val_dice,
        epochs_without_improvement,
    )
    return start_epoch, best_val_dice, epochs_without_improvement, history_rows


def train_and_evaluate(config: RunConfig) -> Path:
    run_dir = Path(config.output_root) / config.run_name
    checkpoint_dir = run_dir / "checkpoints"
    test_dir = run_dir / "test_results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Resolved TN3K root: %s", config.data_root)

    seed_everything(config.seed)

    dataset_summary = summarize_tn3k(config.data_root, config.mask_threshold)
    dataset_summary["selected_fold"] = config.fold
    dataset_summary["network"] = config.network
    dataset_summary["selected_split_counts"] = {
        "train": dataset_summary["official_folds"][f"fold{config.fold}"]["train"],
        "val": dataset_summary["official_folds"][f"fold{config.fold}"]["val"],
        "test": dataset_summary["splits"]["test"]["count"],
    }
    with (run_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_summary, handle, indent=2)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    dataloaders = build_dataloaders(config)
    model = build_model(config, device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = None
    if not config.disable_cosine:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)

    best_checkpoint_path = checkpoint_dir / "best_model.pt"
    history_csv_path = run_dir / "history.csv"
    start_epoch, best_val_dice, epochs_without_improvement, history_rows = maybe_resume_training(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        history_csv_path=history_csv_path,
        logger=logger,
        device=device,
    )

    if start_epoch > config.epochs:
        logger.info(
            "Checkpoint epoch %d is already at or beyond requested total epochs %d. Skipping training.",
            start_epoch - 1,
            config.epochs,
        )

    for epoch in range(start_epoch, config.epochs + 1):
        logger.info("Starting epoch %03d/%03d", epoch, config.epochs)
        train_metrics = epoch_pass(
            model=model,
            loader=dataloaders["train"],
            device=device,
            threshold=config.threshold,
            optimizer=optimizer,
            max_steps=config.max_train_steps,
            epoch=epoch,
            total_epochs=config.epochs,
        )
        val_metrics = epoch_pass(
            model=model,
            loader=dataloaders["val"],
            device=device,
            threshold=config.threshold,
            optimizer=None,
            max_steps=config.max_val_steps,
            epoch=epoch,
            total_epochs=config.epochs,
        )

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_specificity": train_metrics["specificity"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history_rows.append(row)
        write_history_csv(history_rows, history_csv_path)

        logger.info(
            "Epoch %03d | train loss %.4f | train dice %.4f | val loss %.4f | val dice %.4f | val IoU %.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["dice"],
            val_metrics["loss"],
            val_metrics["dice"],
            val_metrics["iou"],
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_dice=best_val_dice,
                epochs_without_improvement=epochs_without_improvement,
                config=config,
            )
            logger.info("Saved new best checkpoint with val dice %.4f", best_val_dice)
        else:
            epochs_without_improvement += 1
            logger.info(
                "No validation improvement. Early-stop counter: %d/%d",
                epochs_without_improvement,
                config.early_stopping,
            )

        save_checkpoint(
            checkpoint_dir / "last_model.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_dice=best_val_dice,
            epochs_without_improvement=epochs_without_improvement,
            config=config,
        )

        if epochs_without_improvement >= config.early_stopping:
            logger.info("Early stopping triggered.")
            break

    if not best_checkpoint_path.is_file():
        fallback_checkpoint = Path(config.resume_from) if config.resume_from else checkpoint_dir / "last_model.pt"
        if fallback_checkpoint.is_file():
            logger.info("Best checkpoint missing. Falling back to: %s", fallback_checkpoint)
            best_checkpoint_path = fallback_checkpoint
        else:
            raise FileNotFoundError(f"No checkpoint available for testing in {checkpoint_dir}")

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_results = evaluate_test_set(
        model=model,
        loader=dataloaders["test"],
        device=device,
        threshold=config.threshold,
        mask_threshold=config.mask_threshold,
        output_dir=test_dir,
        max_samples=config.max_test_samples,
    )
    logger.info("Test mean Dice: %.4f", test_results["summary"]["mean"]["dice"])
    logger.info("Test mean IoU: %.4f", test_results["summary"]["mean"]["iou"])
    logger.info("Saved test artifacts to %s", test_dir)
    return run_dir


def main() -> None:
    args = parse_args()
    config = build_run_config(args)
    run_dir = train_and_evaluate(config)
    print(f"Run completed. Outputs are in: {run_dir}")


if __name__ == "__main__":
    main()
