"""특징 × 분류기 비교 (기획서 7장): DummyClassifier, 로지스틱 회귀, SVM, 랜덤포레스트.

    python -m src.train_classical --features hand
    python -m src.train_classical --features resnet18
    python -m src.train_classical --features hand --feature-group color     # 실험: 색상만
    python -m src.train_classical --features resnet18 --final               # test 최종 평가 (설정 확정 후 1회)

절차 (기획서 8장 누수 방지):
  1. Pipeline(StandardScaler → 분류기)을 train에만 fit. class_weight="balanced".
  2. 하이퍼파라미터는 val Macro-F1로 고른다 (동점이면 오안내율이 낮은 쪽).
  3. 확률 보정(다중 클래스 Platt scaling = 모델 점수에 대한 로지스틱 회귀)도 val로 한다. val 자체의 보정 후 지표·τ 선택에는
     val을 그룹 2-fold로 나눠 서로 다른 절반으로 보정한 out-of-fold 확률을 쓴다 (같은 데이터로 맞추고 평가하지 않기 위해).
  4. τ = val에서 '자동 안내 중 일반쓰레기→재활용 오안내율 ≤ --max-misguide'를 만족하는 가장 낮은 값.
  5. --final 일 때만 test를 열어 한 번 평가한다.
결과: reports/experiments/<run-id>/ (record.json, 그림), 모델 파일은 $ML_TEAMWORK_DATA/../models/ (Git 제외).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import evaluate as ev
from src.dataset import DEFAULT_VERSION, data_root, processed_dir

CLASSES = [ev.RECYCLE, ev.GENERAL, ev.SPECIAL]


def model_grid(name: str, n_features: int, seed: int) -> list[tuple[dict, object]]:
    if name == "dummy":
        return [({"strategy": "most_frequent"}, DummyClassifier(strategy="most_frequent"))]
    if name == "logreg":
        return [({"C": C}, make_pipeline(StandardScaler(), LogisticRegression(
            C=C, class_weight="balanced", max_iter=3000, random_state=seed))) for C in (0.01, 0.1, 1.0)]
    if name == "svm":
        out = []
        for C in (1.0, 10.0):
            for gm in (0.5, 1.0, 2.0):
                gamma = gm / n_features
                out.append(({"C": C, "gamma": round(gamma, 6), "gamma_x_scale": gm}, make_pipeline(
                    StandardScaler(), SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced",
                                          random_state=seed))))
        return out
    if name == "rf":
        return [({"n_estimators": 300, "max_depth": d, "min_samples_leaf": 2}, RandomForestClassifier(
            n_estimators=300, max_depth=d, min_samples_leaf=2, class_weight="balanced", n_jobs=-1,
            random_state=seed)) for d in (10, 20, None)]
    raise ValueError(name)


def _proba(model, X) -> np.ndarray | None:
    try:
        return aligned_proba(model, X)
    except AttributeError:   # SVC(probability=False)
        return None


def _scores(model, X) -> np.ndarray:
    """보정 입력용 점수: decision_function이 있으면 그 값, 없으면(랜덤포레스트) log 확률."""
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return s if s.ndim == 2 else np.column_stack([-s, s])
    return np.log(np.clip(model.predict_proba(X), 1e-6, None))


class PlattCalibrator:
    """다중 클래스 Platt scaling: 학습된 모델의 점수 → 다항 로지스틱 회귀 (val로만 fit).

    CalibratedClassifierCV(method="sigmoid")와 같은 목적이다. 학습된 모델은 다시 학습하지 않는다.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.lr_ = LogisticRegression(C=100.0, max_iter=2000).fit(_scores(self.model, X), y)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(_scores(self.model, X))


def calibrate(model, X, y) -> PlattCalibrator:
    return PlattCalibrator(model).fit(X, y)


def aligned_proba(est, X) -> np.ndarray:
    """est.classes_ 순서를 CLASSES 순서로 맞춘다. 보정용 데이터에 없던 클래스는 확률 0."""
    p = est.predict_proba(X)
    out = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(est.classes_):
        out[:, CLASSES.index(c)] = p[:, j]
    return out


def oof_calibrated_proba(model, X, y, groups, seed: int) -> np.ndarray:
    """val을 그룹 2-fold로 나눠, 한쪽으로 보정하고 다른 쪽을 예측한 확률."""
    proba = np.zeros((len(y), len(CLASSES)))
    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    for fit_idx, pred_idx in sgkf.split(X, y, groups):
        proba[pred_idx] = aligned_proba(calibrate(model, X[fit_idx], y[fit_idx]), X[pred_idx])
    return proba


def model_size(model) -> dict:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if isinstance(est, LogisticRegression):
        return {"parameters": int(est.coef_.size + est.intercept_.size)}
    if isinstance(est, SVC):
        return {"support_vectors": int(est.support_vectors_.shape[0]),
                "parameters": int(est.support_vectors_.size + est.dual_coef_.size)}
    if isinstance(est, RandomForestClassifier):
        return {"total_nodes": int(sum(t.tree_.node_count for t in est.estimators_))}
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=["hand", "resnet18"], required=True)
    ap.add_argument("--feature-group", choices=["color", "texture_shape"], help="hand 특징 중 일부만 사용")
    ap.add_argument("--models", default="dummy,logreg,svm,rf")
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--max-misguide", type=float, default=0.05,
                    help="τ 선택 기준: 자동 안내 중 일반쓰레기→재활용 오안내율 상한 (기획서: 결정 필요, 잠정값)")
    ap.add_argument("--max-train", type=int, help="개발용: train 표본 수 상한")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--final", action="store_true", help="test 최종 평가 (설정 확정 후 1회만)")
    args = ap.parse_args(argv)

    root = processed_dir(args.version)
    man = pd.read_csv(root / "manifest.csv")
    X = np.load(root / f"features_{args.features}.npy")
    fmeta = json.loads((root / f"features_{args.features}.json").read_text(encoding="utf-8"))
    if args.feature_group:
        X = X[:, fmeta["groups"][args.feature_group]]
    split_meta = json.loads((root / "split_meta.json").read_text(encoding="utf-8"))

    sets = {s: np.flatnonzero(man["split"].to_numpy() == s) for s in ("train", "val", "test")}
    if args.max_train and len(sets["train"]) > args.max_train:
        rng = np.random.default_rng(args.seed)
        sets["train"] = np.sort(rng.choice(sets["train"], args.max_train, replace=False))
    y = man["verdict"].to_numpy()
    items = man["item"].to_numpy()
    groups = man["group"].to_numpy()
    tr, va, te = sets["train"], sets["val"], sets["test"]

    feat_name = args.features + (f"-{args.feature_group}" if args.feature_group else "")
    run_id, run_dir = ev.new_run_dir(f"classical-{feat_name}" + ("-final" if args.final else ""))
    model_dir = data_root().parent / "models" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {run_id}: train {len(tr):,} / val {len(va):,} / test {len(te):,}  X={X.shape}")

    results = {}
    for name in args.models.split(","):
        trials = []
        best = None
        for params, est in model_grid(name, X.shape[1], args.seed):
            t0 = time.time()
            m = clone(est).fit(X[tr], y[tr])
            fit_s = time.time() - t0
            pv = m.predict(X[va])
            met = ev.verdict_metrics(y[va], pv, CLASSES)
            trials.append({"params": params, "fit_seconds": round(fit_s, 2),
                           "val_macro_f1": met["macro_f1"], "val_misguide_rate": met["misguide_rate"],
                           "train_macro_f1": ev.verdict_metrics(y[tr], m.predict(X[tr]), CLASSES)["macro_f1"]})
            print(f"  {name} {params}: val F1={met['macro_f1']:.3f} misguide={met['misguide_rate']}"
                  f" ({fit_s:.1f}s)")
            key = (met["macro_f1"], -(met["misguide_rate"] or 0))
            if best is None or key > best[0]:
                best = (key, params, m)
        _, params, model = best

        t0 = time.time()
        pred_va = model.predict(X[va])
        infer_ms = (time.time() - t0) / max(len(va), 1) * 1000
        raw_proba = _proba(model, X[va])
        out = {"selected_params": params, "trials": trials, "size": model_size(model),
               "inference_ms_per_image_features_only": round(infer_ms, 4),
               "val": ev.verdict_metrics(y[va], pred_va, CLASSES, items[va], raw_proba)}

        if name != "dummy":
            oof = oof_calibrated_proba(model, X[va], y[va], groups[va], args.seed)
            pred_cal = np.asarray(CLASSES)[oof.argmax(1)]
            out["val_calibrated_oof"] = ev.verdict_metrics(y[va], pred_cal, CLASSES, items[va], oof)
            curve = ev.selective_curve(y[va], oof, CLASSES)
            tau = ev.pick_tau(curve, args.max_misguide)
            out["val_selective_curve"] = curve
            out["tau"] = tau
            ev.plot_reliability(ev.reliability_bins(raw_proba, y[va], CLASSES) if raw_proba is not None else None,
                                ev.reliability_bins(oof, y[va], CLASSES), run_dir / f"reliability_{name}_val.png",
                                f"{feat_name} + {name} (val)")
            ev.plot_selective(curve, run_dir / f"selective_{name}_val.png", f"{feat_name} + {name} (val, 보정 후)",
                              tau["tau"] if tau else None)
            calibrator = calibrate(model, X[va], y[va])      # test·시연용: val 전체로 보정
            with open(model_dir / f"{name}.pkl", "wb") as f:
                pickle.dump({"model": model, "calibrator": calibrator, "classes": CLASSES,
                             "features": args.features, "feature_group": args.feature_group,
                             "tau": tau["tau"] if tau else None}, f)
        ev.plot_confusion(out["val"]["confusion"], CLASSES, run_dir / f"confusion_{name}_val.png",
                          f"{feat_name} + {name} (val, n={len(va)})")

        if args.final:
            calibrator = calibrate(model, X[va], y[va]) if name != "dummy" else None
            ptp = aligned_proba(calibrator, X[te]) if calibrator else None
            pt = np.asarray(CLASSES)[ptp.argmax(1)] if calibrator else model.predict(X[te])
            out["test"] = ev.verdict_metrics(y[te], pt, CLASSES, items[te], ptp)
            if ptp is not None and out.get("tau"):
                t = out["tau"]["tau"]
                out["test_at_tau"] = ev.selective_curve(y[te], ptp, CLASSES, taus=[t])[0]
            ev.plot_confusion(out["test"]["confusion"], CLASSES, run_dir / f"confusion_{name}_test.png",
                              f"{feat_name} + {name} (test, n={len(te)})")
        results[name] = out
        v = out["val"]
        print(f"→ {name}: val F1={v['macro_f1']:.3f} acc={v['accuracy']:.3f} misguide={v['misguide_rate']}"
              f" battery={v.get('battery_recall')}")

    ev.write_record(run_dir, run_id, {
        "status": "완료",
        "task": "1단계 판정 (재활용/일반쓰레기/별도배출)",
        "input": {"data_version": args.version, "split_meta": split_meta, "features": feat_name,
                  "feature_dim": int(X.shape[1]), "n_train": int(len(tr)), "n_val": int(len(va)),
                  "n_test_evaluated": int(len(te)) if args.final else 0},
        "settings": {"models": args.models, "seed": args.seed, "max_misguide_for_tau": args.max_misguide,
                     "calibration": "multiclass Platt scaling (LogisticRegression on model scores) fit on val; val 지표는 그룹 2-fold OOF",
                     "class_weight": "balanced", "max_train": args.max_train},
        "results": results,
        "model_files": str(model_dir),
    })
    print(f"saved {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
