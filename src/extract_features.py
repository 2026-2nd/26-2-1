"""crop 이미지 → 특징 행렬 (기획서 7장 '사진을 숫자 벡터로 바꾸는 방법 2가지').

    python -m src.extract_features --kind hand
    python -m src.extract_features --kind resnet18

출력: $ML_TEAMWORK_DATA/processed/<version>/features_<kind>.npy (manifest.csv 행 순서와 같음)
      features_<kind>.json (특징 이름·차원·설정)
두 특징 모두 이미지 한 장만으로 계산하므로 train 정보를 쓰지 않는다. 스케일링은 학습 Pipeline에서 한다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import DEFAULT_VERSION, processed_dir


def _read(path: Path):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def hand_features(manifest: pd.DataFrame, root: Path) -> tuple[np.ndarray, dict]:
    from src import features
    X = np.zeros((len(manifest), len(features.feature_names())), np.float32)
    for i, r in enumerate(manifest.itertuples()):
        X[i] = features.extract(_read(root / r.crop_path), r.w, r.h)
        if (i + 1) % 2000 == 0:
            print(f"  hand {i + 1:,}/{len(manifest):,}")
    meta = {"names": features.feature_names(),
            "groups": {k: [features.feature_names().index(n) for n in v] for k, v in features.FEATURE_GROUPS.items()}}
    return X, meta


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resnet18_backbone():
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    return m.eval()


def resnet_features(manifest: pd.DataFrame, root: Path, batch: int = 64) -> tuple[np.ndarray, dict]:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet18_backbone().to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = np.zeros((len(manifest), 512), np.float32)
    paths = [root / p for p in manifest["crop_path"]]
    with torch.no_grad():
        for s in range(0, len(paths), batch):
            imgs = np.stack([_read(p)[:, :, ::-1] for p in paths[s:s + batch]])   # BGR → RGB, 224×224
            x = torch.from_numpy(imgs.copy()).to(device).permute(0, 3, 1, 2).float() / 255.0
            out[s:s + len(imgs)] = model((x - mean) / std).cpu().numpy()
            if (s // batch) % 20 == 0:
                print(f"  resnet18 {s + len(imgs):,}/{len(paths):,} ({device})")
    return out, {"names": [f"resnet18_{i}" for i in range(512)], "weights": "IMAGENET1K_V1", "device": device}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["hand", "resnet18"], required=True)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    args = ap.parse_args(argv)
    root = processed_dir(args.version)
    manifest = pd.read_csv(root / "manifest.csv")
    t0 = time.time()
    X, meta = hand_features(manifest, root) if args.kind == "hand" else resnet_features(manifest, root)
    meta.update({"kind": args.kind, "n": len(X), "dim": int(X.shape[1]), "seconds": round(time.time() - t0, 1)})
    np.save(root / f"features_{args.kind}.npy", X)
    with open(root / f"features_{args.kind}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"saved features_{args.kind}.npy {X.shape} in {meta['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
