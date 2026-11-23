"""ResNet18 미세조정 (기획서 7장 딥러닝 선택 과제): 마지막 블록(layer4) + 분류층만 학습.

    python -m src.finetune --epochs 8                 # Colab GPU 권장
    python -m src.finetune --epochs 8 --final         # test 최종 평가 (설정 확정 후 1회)

- 증강(RandomResizedCrop·뒤집기·색상·회전)은 train에만 적용한다.
- 클래스 가중치(balanced)를 손실에 넣는다.
- 에폭 선택은 val Macro-F1 기준. 학습 곡선을 저장한다.
- 확률 보정은 temperature scaling [Guo 2017]을 val로 맞춘다. val 지표·τ는 그룹 2-fold OOF 온도로 계산한다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate as ev
from src.dataset import DEFAULT_VERSION, data_root, processed_dir
from src.extract_features import IMAGENET_MEAN, IMAGENET_STD

CLASSES = [ev.RECYCLE, ev.GENERAL, ev.SPECIAL]


def build_model(n_classes: int = 3, train_blocks: tuple[str, ...] = ("layer4",)):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    for b in train_blocks:
        for p in getattr(m, b).parameters():
            p.requires_grad = True
    m.fc = torch.nn.Linear(m.fc.in_features, n_classes)
    return m


def make_loaders(man: pd.DataFrame, root: Path, idx: dict, batch: int, workers: int, seed: int):
    import torch
    from PIL import Image
    from torchvision import transforms as T

    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    tf_train = T.Compose([T.RandomResizedCrop(224, scale=(0.75, 1.0)), T.RandomHorizontalFlip(),
                          T.RandomRotation(10), T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), norm])
    tf_eval = T.Compose([T.ToTensor(), norm])
    y_idx = man["verdict"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    class DS(torch.utils.data.Dataset):
        def __init__(self, rows, tf):
            self.rows, self.tf = rows, tf

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, k):
            i = self.rows[k]
            img = Image.open(root / man["crop_path"].iat[i]).convert("RGB")
            return self.tf(img), int(y_idx[i])

    g = torch.Generator().manual_seed(seed)
    mk = lambda rows, tf, shuffle: torch.utils.data.DataLoader(  # noqa: E731
        DS(rows, tf), batch_size=batch, shuffle=shuffle, num_workers=workers, generator=g if shuffle else None)
    return {"train": mk(idx["train"], tf_train, True), "val": mk(idx["val"], tf_eval, False),
            "test": mk(idx["test"], tf_eval, False)}, y_idx


def predict_logits(model, loader, device) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """val NLL을 최소화하는 온도 T (1차원 탐색)."""
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.25), np.log(8.0), 200)):
        p = softmax(logits / t)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def oof_temperature_proba(logits, y, groups, seed) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold
    proba = np.zeros_like(logits)
    for a, b in StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed).split(logits, y, groups):
        proba[b] = softmax(logits[b] / fit_temperature(logits[a], y[a]))
    return proba


def plot_curves(hist: list[dict], path: Path) -> None:
    plt = ev._plt()
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    e = [h["epoch"] for h in hist]
    ax[0].plot(e, [h["train_loss"] for h in hist], marker="o", label="train")
    ax[0].plot(e, [h["val_loss"] for h in hist], marker="o", label="val")
    ax[0].set_title("loss")
    ax[1].plot(e, [h["train_macro_f1"] for h in hist], marker="o", label="train (증강)")
    ax[1].plot(e, [h["val_macro_f1"] for h in hist], marker="o", label="val")
    ax[1].set_title("Macro-F1")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-train", type=int, help="개발용: train 표본 수 상한")
    ap.add_argument("--max-misguide", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = processed_dir(args.version)
    man = pd.read_csv(root / "manifest.csv")
    idx = {s: np.flatnonzero(man["split"].to_numpy() == s) for s in ("train", "val", "test")}
    if args.max_train and len(idx["train"]) > args.max_train:
        idx["train"] = np.sort(np.random.default_rng(args.seed).choice(idx["train"], args.max_train, replace=False))
    loaders, y_idx = make_loaders(man, root, idx, args.batch, args.workers, args.seed)
    y_va = y_idx[idx["val"]]

    model = build_model().to(device)
    counts = np.bincount(y_idx[idx["train"]], minlength=len(CLASSES))
    weights = torch.tensor(len(idx["train"]) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32,
                           device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW([{"params": model.fc.parameters(), "lr": args.lr_head},
                             {"params": model.layer4.parameters(), "lr": args.lr_backbone}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    run_id, run_dir = ev.new_run_dir("finetune-resnet18" + ("-final" if args.final else ""))
    model_dir = data_root().parent / "models" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    hist, best = [], None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, tp, tt = 0.0, [], []
        for x, yb in loaders["train"]:
            x, yb = x.to(device), yb.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            tp.append(out.argmax(1).cpu().numpy())
            tt.append(yb.cpu().numpy())
        sched.step()
        lv = predict_logits(model, loaders["val"], device)
        val_loss = float(loss_fn(torch.tensor(lv, device=device), torch.tensor(y_va, device=device)).item())
        C = np.asarray(CLASSES)
        vm = ev.verdict_metrics(C[y_va], C[lv.argmax(1)], CLASSES)
        trm = ev.verdict_metrics(C[np.concatenate(tt)], C[np.concatenate(tp)], CLASSES)
        h = {"epoch": epoch, "train_loss": tl / len(idx["train"]), "val_loss": val_loss,
             "train_macro_f1": trm["macro_f1"], "val_macro_f1": vm["macro_f1"],
             "val_misguide_rate": vm["misguide_rate"], "seconds": round(time.time() - t_start, 1)}
        hist.append(h)
        print(json.dumps(h, ensure_ascii=False))
        if best is None or vm["macro_f1"] > best[0]:
            best = (vm["macro_f1"], epoch)
            torch.save(model.state_dict(), model_dir / "best.pt")
    plot_curves(hist, run_dir / "learning_curve.png")

    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device))
    C = np.asarray(CLASSES)
    lv = predict_logits(model, loaders["val"], device)
    items, groups = man["item"].to_numpy(), man["group"].to_numpy()
    t0 = time.time()
    _ = predict_logits(model, loaders["val"], device)
    infer_ms = (time.time() - t0) / max(len(y_va), 1) * 1000
    T = fit_temperature(lv, y_va)
    oof = oof_temperature_proba(lv, y_va, groups[idx["val"]], args.seed)
    curve = ev.selective_curve(C[y_va], oof, CLASSES)
    tau = ev.pick_tau(curve, args.max_misguide)
    res = {
        "best_epoch": best[1], "history": hist, "temperature": T,
        "size": {"parameters": int(sum(p.numel() for p in model.parameters())),
                 "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad))},
        "inference_ms_per_image_incl_loading": round(infer_ms, 3), "device": device,
        "val": ev.verdict_metrics(C[y_va], C[lv.argmax(1)], CLASSES, items[idx["val"]], softmax(lv)),
        "val_calibrated_oof": ev.verdict_metrics(C[y_va], C[oof.argmax(1)], CLASSES, items[idx["val"]], oof),
        "val_selective_curve": curve, "tau": tau,
    }
    ev.plot_confusion(res["val"]["confusion"], CLASSES, run_dir / "confusion_val.png",
                      f"ResNet18 미세조정 (val, n={len(y_va)})")
    ev.plot_reliability(ev.reliability_bins(softmax(lv), C[y_va], CLASSES),
                        ev.reliability_bins(oof, C[y_va], CLASSES), run_dir / "reliability_val.png",
                        "ResNet18 미세조정 (val)")
    ev.plot_selective(curve, run_dir / "selective_val.png", "ResNet18 미세조정 (val, 보정 후)",
                      tau["tau"] if tau else None)
    if args.final:
        lt = predict_logits(model, loaders["test"], device)
        y_te = y_idx[idx["test"]]
        pt = softmax(lt / T)
        res["test"] = ev.verdict_metrics(C[y_te], C[pt.argmax(1)], CLASSES, items[idx["test"]], pt)
        if tau:
            res["test_at_tau"] = ev.selective_curve(C[y_te], pt, CLASSES, taus=[tau["tau"]])[0]
        ev.plot_confusion(res["test"]["confusion"], CLASSES, run_dir / "confusion_test.png",
                          f"ResNet18 미세조정 (test, n={len(y_te)})")
    with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"classes": CLASSES, "temperature": T, "tau": tau["tau"] if tau else None}, f, ensure_ascii=False)
    ev.write_record(run_dir, run_id, {
        "status": "완료", "task": "1단계 판정 (재활용/일반쓰레기/별도배출)",
        "input": {"data_version": args.version,
                  "split_meta": json.loads((root / "split_meta.json").read_text(encoding="utf-8")),
                  "n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])),
                  "n_test_evaluated": int(len(idx["test"])) if args.final else 0},
        "settings": {k: v for k, v in vars(args).items()} | {
            "trainable": "layer4 + fc", "augment": "RandomResizedCrop(0.75-1)+HFlip+Rot10+ColorJitter (train only)",
            "loss": "CrossEntropy(class weight balanced)", "optimizer": "AdamW + Cosine",
            "calibration": "temperature scaling on val; val 지표는 그룹 2-fold OOF"},
        "results": res, "model_files": str(model_dir),
    })
    print(f"saved {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
