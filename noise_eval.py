"""Robustness under simulated ultrasound degradations (speckle noise, Gaussian blur,
contrast reduction), evaluated on FIXED trained checkpoints (no re-training) — mirrors
the robustness protocol of the DSANet paper. Reuses the E0 build_model + dataset so the
clean numbers reproduce the reported std@0.5 Dice.

Usage:
  python noise_eval.py --run-dir <run> --data-root <root> --degradation speckle --severity severe --out <json>
"""
import argparse, json, sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_e0_honest import build_model, RunConfig, numpy_metrics
from dataset_trainval import TN3KDataset

# Severity presets (severe ~ the "severe" column of DSANet Table V).
PRESETS = {
    "mild":   {"speckle": 0.10, "blur": (7, 1.5), "contrast": 0.70},
    "severe": {"speckle": 0.20, "blur": (9, 2.0), "contrast": 0.35},
}


def degrade(x, kind, sev):
    """x: (B,3,H,W) float in [0,1]. Returns degraded clone in [0,1]."""
    p = PRESETS[sev]
    if kind == "clean":
        return x
    if kind == "speckle":                    # multiplicative speckle: x*(1+n)
        n = torch.randn_like(x) * p["speckle"]
        return (x + x * n).clamp(0, 1)
    if kind == "blur":
        k, sigma = p["blur"]
        return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])
    if kind == "contrast":                    # pull toward per-image mean
        mean = x.mean(dim=(2, 3), keepdim=True)
        return ((x - mean) * p["contrast"] + mean).clamp(0, 1)
    raise ValueError(kind)


def load_config(run_dir):
    cfg = json.load(open(Path(run_dir) / "config.json"))
    keep = {f.name for f in fields(RunConfig)}
    kw = {k: v for k, v in cfg.items() if k in keep}
    if "data_root" in kw:
        kw["data_root"] = Path(kw["data_root"])
    # backfill any required field added after some older runs were serialised
    import dataclasses
    known = {("ablate", ""), ("freeze", "none"), ("no_pretrain", False),
             ("normalize_ds", False), ("sweep_only", False)}
    for name, default in known:
        kw.setdefault(name, default)
    for f in fields(RunConfig):
        if f.name not in kw and f.default is dataclasses.MISSING and \
           f.default_factory is dataclasses.MISSING:  # still-missing required field
            kw[f.name] = "" if f.type in ("str", str) else 0
    return RunConfig(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--degradation", required=True,
                    choices=["clean", "speckle", "blur", "contrast", "all"])
    ap.add_argument("--severity", default="severe", choices=["mild", "severe"])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--token", default=None, help="display/output name (for --degradation all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(a.run_dir)
    config.image_size = a.image_size
    model = build_model(config, device)
    ckpt = torch.load(Path(a.run_dir) / "checkpoints" / "best_model.pt", map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    ds = TN3KDataset(a.data_root, "test", a.image_size, augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    degs = ["clean", "speckle", "blur", "contrast"] if a.degradation == "all" else [a.degradation]
    out_dir = Path(a.out) if a.degradation == "all" else Path(a.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(a.run_dir).name if a.degradation == "all" else None
    token = a.token or Path(a.run_dir).name

    keys = ["dice", "iou", "precision", "recall"]
    for deg in degs:
        torch.manual_seed(a.seed)                    # same noise draw regardless of order
        records = []
        with torch.inference_mode():
            for batch in loader:
                imgs = degrade(batch["image"].to(device), deg, a.severity)
                out = model(imgs)
                logits = out[0] if isinstance(out, (list, tuple)) else out
                probs = torch.sigmoid(logits).cpu().numpy()
                for bi in range(imgs.shape[0]):
                    gt = (np.asarray(Image.open(Path(batch["mask_path"][bi])).convert("L"), np.uint8) > 127).astype(np.uint8)
                    oh, ow = gt.shape
                    pm = (probs[bi, 0] * 255).astype(np.uint8)
                    resized = np.asarray(Image.fromarray(pm).resize((ow, oh), Image.Resampling.BILINEAR), np.uint8)
                    pred = (resized >= int(a.threshold * 255)).astype(np.uint8)
                    records.append(numpy_metrics(pred, gt))
        summary = {"degradation": deg, "severity": a.severity, "n": len(records),
                   "mean": {k: float(np.mean([r[k] for r in records])) for k in keys},
                   "std":  {k: float(np.std([r[k] for r in records])) for k in keys}}
        outp = (out_dir / f"{token}_{deg}.json") if a.degradation == "all" else Path(a.out)
        json.dump(summary, open(outp, "w"), indent=2)
        print(f"{token:14} {deg:9} {a.severity:6} Dice {summary['mean']['dice']:.4f}  IoU {summary['mean']['iou']:.4f}", flush=True)


if __name__ == "__main__":
    main()
