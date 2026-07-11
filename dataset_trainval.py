"""TN3K dataset with 'trainval' split support (all 2879 images, no fold filtering)."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
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


def sorted_image_names(image_dir: Path) -> List[str]:
    return sorted(
        [p.name for p in image_dir.iterdir() if p.is_file()],
        key=lambda n: int(Path(n).stem),
    )


def resolve_tn3k_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    if (root / "raw").is_dir():
        root = root / "raw"
    for d in ("trainval-image", "trainval-mask", "test-image", "test-mask"):
        if not (root / d).is_dir():
            raise FileNotFoundError(f"Missing: {root / d}")
    return root


def build_samples(root: Path, split: str, fold: int = 0) -> List[TN3KSample]:
    if split == "test":
        names = sorted_image_names(root / "test-image")
        img_dir, msk_dir = root / "test-image", root / "test-mask"
    elif split == "trainval":
        # ALL trainval images — no fold filtering
        names = sorted_image_names(root / "trainval-image")
        img_dir, msk_dir = root / "trainval-image", root / "trainval-mask"
    elif split in ("train", "val"):
        all_names = sorted_image_names(root / "trainval-image")
        with (root / f"tn3k-trainval-fold{fold}.json").open() as f:
            indices = json.load(f)[split]
        names = [all_names[i] for i in indices]
        img_dir, msk_dir = root / "trainval-image", root / "trainval-mask"
    else:
        raise ValueError(f"Unsupported split: {split}")

    samples = []
    for name in names:
        s = TN3KSample(Path(name).stem, img_dir / name, msk_dir / name)
        if not s.image_path.is_file():
            raise FileNotFoundError(f"Missing image: {s.image_path}")
        if not s.mask_path.is_file():
            raise FileNotFoundError(f"Missing mask: {s.mask_path}")
        samples.append(s)
    return samples


class TN3KDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str, image_size: int,
                 fold: int = 0, augment: bool = False, mask_threshold: int = MASK_THRESHOLD,
                 noise_aug: float = 0.0):
        self.root = resolve_tn3k_root(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.mask_threshold = mask_threshold
        self.noise_aug = noise_aug          # >0: random speckle/blur/contrast on the image tensor
        self.samples = build_samples(self.root, split, fold)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        rot = random.randint(0, 3)
        if rot:
            image = np.rot90(image, k=rot).copy()
            mask = np.rot90(mask, k=rot).copy()
        return image, mask

    def _degrade(self, img_t: torch.Tensor) -> torch.Tensor:
        """Randomly apply one of speckle / blur / contrast to a (3,H,W) [0,1] image.
        Matches the evaluation degradations so the model learns to tolerate them."""
        r = random.random()
        if r < 0.5:                                   # no degradation half the time
            return img_t
        kind = random.choice(["speckle", "blur", "contrast"])
        if kind == "speckle":
            sigma = random.uniform(0.05, self.noise_aug)
            noise = torch.randn_like(img_t) * sigma
            img_t = (img_t + img_t * noise).clamp(0, 1)
        elif kind == "blur":
            k = random.choice([3, 5, 7]); sig = random.uniform(0.5, 2.0)
            img_t = TF.gaussian_blur(img_t, kernel_size=[k, k], sigma=[sig, sig])
        else:                                          # contrast
            c = random.uniform(0.4, 1.0)
            m = img_t.mean(dim=(1, 2), keepdim=True)
            img_t = ((img_t - m) * c + m).clamp(0, 1)
        return img_t

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        image = np.asarray(Image.open(s.image_path).convert("RGB"))
        mask = (np.asarray(Image.open(s.mask_path).convert("L"), dtype=np.uint8) > self.mask_threshold).astype(np.uint8)
        oh, ow = image.shape[:2]

        if self.augment:
            image, mask = self._augment(image, mask)

        img_pil = TF.resize(Image.fromarray(image), [self.image_size, self.image_size],
                             interpolation=InterpolationMode.BILINEAR, antialias=True)
        msk_pil = TF.resize(Image.fromarray(mask * 255), [self.image_size, self.image_size],
                             interpolation=InterpolationMode.NEAREST)
        img_t = TF.to_tensor(img_pil)
        msk_t = (TF.to_tensor(msk_pil) > 0.5).float()

        # random image degradation augmentation (speckle / blur / contrast) — image only
        if self.augment and self.noise_aug > 0:
            img_t = self._degrade(img_t)

        return {
            "image": img_t, "mask": msk_t,
            "sample_id": s.sample_id, "image_path": str(s.image_path),
            "mask_path": str(s.mask_path),
            "original_size": torch.tensor([oh, ow], dtype=torch.int64),
        }
