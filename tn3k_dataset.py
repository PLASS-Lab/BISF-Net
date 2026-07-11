from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


MASK_THRESHOLD = 127


@dataclass(frozen=True)
class TN3KSample:
    sample_id: str
    image_path: Path
    mask_path: Path


def default_tn3k_root() -> Path:
    return Path(__file__).resolve().parent.parent / "Datasets" / "TN3K" / "raw"


def resolve_tn3k_root(data_root: str | Path | None = None) -> Path:
    root = Path(data_root).expanduser().resolve() if data_root else default_tn3k_root().resolve()
    required_dirs = [
        root / "trainval-image",
        root / "trainval-mask",
        root / "test-image",
        root / "test-mask",
    ]
    missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            "TN3K dataset is incomplete. Missing directories: " + ", ".join(missing_dirs)
        )
    return root


def sorted_image_names(image_dir: Path) -> List[str]:
    return sorted(
        [path.name for path in image_dir.iterdir() if path.is_file()],
        key=lambda name: int(Path(name).stem),
    )


def load_official_fold(root: Path, fold: int) -> Dict[str, List[int]]:
    split_file = root / f"tn3k-trainval-fold{fold}.json"
    if not split_file.is_file():
        raise FileNotFoundError(f"Expected official split file at {split_file}")
    with split_file.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    return {"train": split["train"], "val": split["val"]}


def build_samples(root: Path, split: str, fold: int) -> List[TN3KSample]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    if split == "test":
        names = sorted_image_names(root / "test-image")
        image_dir = root / "test-image"
        mask_dir = root / "test-mask"
    else:
        trainval_names = sorted_image_names(root / "trainval-image")
        fold_indices = load_official_fold(root, fold)[split]
        names = [trainval_names[index] for index in fold_indices]
        image_dir = root / "trainval-image"
        mask_dir = root / "trainval-mask"

    samples: List[TN3KSample] = []
    for name in names:
        sample = TN3KSample(
            sample_id=Path(name).stem,
            image_path=image_dir / name,
            mask_path=mask_dir / name,
        )
        if not sample.image_path.is_file():
            raise FileNotFoundError(f"Missing image: {sample.image_path}")
        if not sample.mask_path.is_file():
            raise FileNotFoundError(f"Missing mask: {sample.mask_path}")
        samples.append(sample)
    return samples


def _load_image(sample: TN3KSample) -> np.ndarray:
    return np.asarray(Image.open(sample.image_path).convert("RGB"))


def _load_mask(sample: TN3KSample, mask_threshold: int = MASK_THRESHOLD) -> np.ndarray:
    mask = np.asarray(Image.open(sample.mask_path).convert("L"), dtype=np.uint8)
    return (mask > mask_threshold).astype(np.uint8)


class TN3KSegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        image_size: int,
        fold: int = 0,
        augment: bool = False,
        mask_threshold: int = MASK_THRESHOLD,
    ) -> None:
        self.root = resolve_tn3k_root(data_root)
        self.split = split
        self.image_size = image_size
        self.fold = fold
        self.augment = augment
        self.mask_threshold = mask_threshold
        self.samples = build_samples(self.root, split, fold)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        rotations = random.randint(0, 3)
        if rotations:
            image = np.rot90(image, k=rotations).copy()
            mask = np.rot90(mask, k=rotations).copy()
        return image, mask

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image = _load_image(sample)
        mask = _load_mask(sample, self.mask_threshold)
        original_height, original_width = image.shape[0], image.shape[1]

        if self.augment:
            image, mask = self._augment(image, mask)

        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask * 255)
        image_pil = TF.resize(
            image_pil,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask_pil = TF.resize(
            mask_pil,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        image_tensor = TF.to_tensor(image_pil)
        mask_tensor = (TF.to_tensor(mask_pil) > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
            "original_size": torch.tensor([original_height, original_width], dtype=torch.int64),
        }


def summarize_tn3k(root: str | Path, mask_threshold: int = MASK_THRESHOLD) -> Dict[str, object]:
    data_root = resolve_tn3k_root(root)
    summary: Dict[str, object] = {
        "dataset_name": "TN3K",
        "dataset_root": str(data_root),
        "mask_threshold": mask_threshold,
        "splits": {},
        "official_folds": {},
    }

    for fold in range(5):
        fold_split = load_official_fold(data_root, fold)
        summary["official_folds"][f"fold{fold}"] = {
            "train": len(fold_split["train"]),
            "val": len(fold_split["val"]),
        }

    total_images = 0
    for split in ["trainval", "test"]:
        image_dir = data_root / f"{split}-image"
        mask_dir = data_root / f"{split}-mask"
        image_names = sorted_image_names(image_dir)
        mask_names = sorted_image_names(mask_dir)
        if image_names != mask_names:
            raise ValueError(f"Image/mask file names do not match for split '{split}'")

        widths: List[int] = []
        heights: List[int] = []
        foreground_ratios: List[float] = []
        mask_midrange_images = 0
        for name in image_names:
            image_size = Image.open(image_dir / name).size
            mask = np.asarray(Image.open(mask_dir / name).convert("L"), dtype=np.uint8)
            widths.append(image_size[0])
            heights.append(image_size[1])
            foreground_ratios.append(float((mask > mask_threshold).mean()))
            mask_midrange_images += int(np.any((mask > 32) & (mask < 223)))

        total_images += len(image_names)
        summary["splits"][split] = {
            "count": len(image_names),
            "width_min": min(widths),
            "width_max": max(widths),
            "height_min": min(heights),
            "height_max": max(heights),
            "unique_size_count": len(set(zip(widths, heights))),
            "mean_foreground_ratio": float(np.mean(foreground_ratios)),
            "median_foreground_ratio": float(median(foreground_ratios)),
            "images_with_midrange_mask_values": mask_midrange_images,
        }

    summary["total_images"] = total_images
    return summary
