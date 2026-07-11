"""Evaluate a trained BiSF-Net checkpoint on a test set (reproduces the paper numbers).

  python evaluate.py --checkpoint checkpoints/bisfnet_tn3k.pt --data-root /path/to/TN3K            # std@0.5
  python evaluate.py --checkpoint checkpoints/bisfnet_tn3k.pt --data-root /path/to/TN3K --tta      # test-time augmentation
"""
import argparse, json
from dataclasses import fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_e0_honest import build_model, RunConfig, evaluate_test_set
from dataset_trainval import TN3KDataset


def load_cfg(config_json):
    c = json.load(open(config_json))
    kw = {f.name: c[f.name] for f in fields(RunConfig) if f.name in c}
    for k, v in (("noise_aug", 0.0), ("train_split", "train"), ("val_split", "val")):
        kw.setdefault(k, v)
    return RunConfig(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True, help="dataset root (TN3K or DDTI layout)")
    ap.add_argument("--config", default="checkpoints/bisfnet_tn3k_config.json")
    ap.add_argument("--output-dir", default="eval_out")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true", help="8-way flip/rotate x multi-scale best-ensemble TTA")
    ap.add_argument("--image-size", type=int, default=224)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_cfg(a.config); cfg.image_size = a.image_size
    model = build_model(cfg, device)
    ck = torch.load(a.checkpoint, map_location=device)
    model.load_state_dict(ck.get("model_state", ck), strict=False)
    model.eval().to(device)

    ds = TN3KDataset(a.data_root, "test", a.image_size, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    res = evaluate_test_set(model, loader, device, a.threshold, 127, Path(a.output_dir),
                            a.image_size, use_best_ensemble=a.tta)
    s = res["summary"]["mean"]
    tag = "TTA (best-ensemble)" if a.tta else f"std@{a.threshold}"
    print(f"\n{ds.__len__()} test images | {tag}")
    print(f"  Dice {s['dice']:.4f}  IoU {s['iou']:.4f}  Precision {s['precision']:.4f}  Recall {s['recall']:.4f}")


if __name__ == "__main__":
    main()
