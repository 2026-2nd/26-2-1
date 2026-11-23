"""공통 평가 지표와 실험 기록 (기획서 7장 '평가 지표', CONTRACTS.md 3절 형식).

1단계 판정(재활용 / 일반쓰레기 / 별도배출) 기준 지표:
    misguide_rate       핵심. 실제 일반쓰레기 중 '재활용'으로 예측한 비율. 낮을수록 좋음.
    macro_f1            주 지표. 세 판정의 F1 단순 평균. 높을수록 좋음.
    accuracy            정확도.
    battery_recall      실제 품목이 건전지인 물건 중 '별도배출'로 예측한 비율.
    item_contam_f1      품목별로, '일반쓰레기' 판정을 양성으로 본 이진 F1 (오염·복합 재질 판별력).
    ece                 top-label 기대 보정 오차 (15 bins). 낮을수록 확률을 믿을 수 있음.
보류 (τ):
    1순위 확률 < τ 이면 보류. coverage = 자동 안내 비율, auto_accuracy = 자동 안내한 것의 정확도,
    auto_misguide_rate = 자동 안내한 것 중 '일반쓰레기 → 재활용' 비율 (분모: 자동 안내된 실제 일반쓰레기).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
RECYCLE, GENERAL, SPECIAL = "재활용", "일반쓰레기", "별도배출"


def misguide_rate(y_true, y_pred) -> float | None:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = y_true == GENERAL
    return float((y_pred[m] == RECYCLE).mean()) if m.any() else None


def ece_score(proba: np.ndarray, y_true, classes, n_bins: int = 15) -> float:
    classes = np.asarray(classes)
    conf = proba.max(axis=1)
    correct = classes[proba.argmax(axis=1)] == np.asarray(y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(proba: np.ndarray, y_true, classes, n_bins: int = 15) -> list[dict]:
    classes = np.asarray(classes)
    conf = proba.max(axis=1)
    correct = classes[proba.argmax(axis=1)] == np.asarray(y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "confidence": float(conf[m].mean()) if m.any() else None,
                    "accuracy": float(correct[m].mean()) if m.any() else None})
    return out


def verdict_metrics(y_true, y_pred, classes, items=None, proba=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "n": int(len(y_true)),
        "misguide_rate": misguide_rate(y_true, y_pred),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": dict(zip(classes, map(float, f1_score(y_true, y_pred, labels=classes, average=None,
                                                                zero_division=0)))),
        "support": {c: int((y_true == c).sum()) for c in classes},
        "confusion": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }
    if items is not None:
        items = np.asarray(items)
        bat = items == "건전지"
        out["battery_recall"] = float((y_pred[bat] == SPECIAL).mean()) if bat.any() else None
        contam = {}
        for it in sorted(set(items)):
            m = items == it
            t, p = y_true[m] == GENERAL, y_pred[m] == GENERAL
            contam[it] = float(f1_score(t, p, zero_division=0)) if t.any() else None
        out["item_contam_f1"] = contam
    if proba is not None:
        out["ece"] = ece_score(proba, y_true, classes)
    return out


def selective_curve(y_true, proba: np.ndarray, classes, taus=None) -> list[dict]:
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    taus = np.round(np.arange(0.50, 0.951, 0.05), 2) if taus is None else taus   # 기획서: τ 50~90%
    rows = []
    for t in taus:
        keep = conf >= t
        rows.append({
            "tau": float(t), "coverage": float(keep.mean()),
            "auto_accuracy": float((pred[keep] == y_true[keep]).mean()) if keep.any() else None,
            "auto_misguide_rate": misguide_rate(y_true[keep], pred[keep]) if keep.any() else None,
        })
    return rows


def pick_tau(curve: list[dict], max_misguide: float) -> dict | None:
    """auto_misguide_rate ≤ 목표를 만족하는 가장 낮은 τ (= 자동 안내 비율이 가장 높음). validation에서만 사용."""
    ok = [r for r in curve if r["auto_misguide_rate"] is not None and r["auto_misguide_rate"] <= max_misguide]
    return min(ok, key=lambda r: r["tau"]) if ok else None


# ── 그림 ─────────────────────────────────────────────────────────────
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        from matplotlib import font_manager
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_confusion(cm, classes, path: Path, title: str) -> None:
    plt = _plt()
    cm = np.asarray(cm, float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    ax.set_title(title, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="행 기준 비율")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability(bins_before, bins_after, path: Path, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1, label="완벽한 보정")
    for bins, lab in ((bins_before, "보정 전"), (bins_after, "보정 후")):
        if bins:
            xs = [b["confidence"] for b in bins if b["n"]]
            ys = [b["accuracy"] for b in bins if b["n"]]
            ax.plot(xs, ys, marker="o", label=lab)
    ax.set_xlabel("예측 확률 (1순위)")
    ax.set_ylabel("실제 정답률")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_selective(curve, path: Path, title: str, tau: float | None = None) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    t = [r["tau"] for r in curve]
    ax.plot(t, [r["coverage"] for r in curve], marker="o", label="자동 안내 비율")
    ax.plot(t, [r["auto_accuracy"] for r in curve], marker="s", label="자동 안내 정확도")
    ax.plot(t, [r["auto_misguide_rate"] for r in curve], marker="^", label="오안내율 (일반→재활용)")
    if tau is not None:
        ax.axvline(tau, c="red", ls=":", label=f"선택 τ={tau:.2f}")
    ax.set_xlabel("보류 기준 τ")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── 실행 기록 ─────────────────────────────────────────────────────────
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def environment_info() -> dict:
    pkgs = {}
    for m in ("numpy", "pandas", "sklearn", "cv2", "skimage", "torch", "torchvision"):
        try:
            pkgs[m] = __import__(m).__version__
        except Exception:
            pass
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pkgs,
            "cpu": platform.processor() or platform.machine()}
    try:
        import torch
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info


def new_run_dir(name: str, root: Path | None = None) -> tuple[str, Path]:
    root = root or REPO_ROOT / "reports" / "experiments"
    day = _dt.date.today().isoformat()
    i = 1
    while (root / f"{day}-{name}-{i:02d}").exists():
        i += 1
    run_id = f"{day}-{name}-{i:02d}"
    path = root / run_id
    path.mkdir(parents=True)
    return run_id, path


def write_record(run_dir: Path, run_id: str, record: dict) -> None:
    """CONTRACTS.md 3절 항목을 담은 record.json. 숫자는 실제 실행 결과만 넣는다."""
    base = {
        "run_id": run_id,
        "author": os.environ.get("ML_TEAMWORK_AUTHOR") or _git("config", "user.name") or "unknown",
        "executed_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "code_commit": _git("rev-parse", "HEAD"),
        "uncommitted_changes": bool(_git("status", "--porcelain", "--", "src", "configs")),
        "command": " ".join(sys.argv),
        "environment": environment_info(),
    }
    base.update(record)
    with open(run_dir / "record.json", "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2, default=float)
