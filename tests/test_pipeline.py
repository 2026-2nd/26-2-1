"""데이터 분할·crop·지표 테스트 (실제 데이터 없이 실행).

실행: python -m unittest discover -s tests -v
"""
import json
import unittest

import numpy as np
import pandas as pd

from src import dataset as ds
from src import evaluate as ev


class LabelConfigTest(unittest.TestCase):
    def test_29_codes_map_to_known_classes(self):
        cfg = ds.load_label_config()
        self.assertEqual(len(cfg["codes"]), 29)
        for code, lab in cfg["codes"].items():
            self.assertIn(lab["verdict"], cfg["verdicts"], code)
            self.assertIn(lab["item"], cfg["items"], code)
            if lab["verdict"] == "일반쓰레기":
                self.assertIn(lab["reason"], cfg["reasons"], code)
            else:
                self.assertIsNone(lab["reason"], code)

    def test_counts_per_verdict_match_plan(self):
        # 기획서 6장 표: 재활용 12 / 이물질 8 / 다중포장재 5 / 별도 4
        cfg = ds.load_label_config()["codes"].values()
        c = pd.Series([(v["verdict"], v["reason"]) for v in cfg]).value_counts()
        self.assertEqual(c[("재활용", None)], 12)
        self.assertEqual(c[("일반쓰레기", "이물질")], 8)
        self.assertEqual(c[("일반쓰레기", "다중포장재")], 5)
        self.assertEqual(c[("별도배출", None)], 4)


class ParseLabelTest(unittest.TestCase):
    def test_gps_and_address_are_not_kept(self):
        data = {"objects": [{"class_name": "c_6_01", "annotation_type": "box",
                             "annotation": {"coord": {"x": 1, "y": 2, "width": 50, "height": 60}}}],
                "Image": "B2_20220721_000002.jpg",
                "Info": {"GPS": "36.1/128.4", "ADDRESS": "어딘가", "RESOLUTION": "1280/960", "METHOD": "B2"}}
        rows = ds.parse_label(data, "B2_20220721_000002.json", "실내형분류기", ds.load_label_config()["codes"])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["verdict"], r["reason"], r["item"]), ("일반쓰레기", "이물질", "플라스틱"))
        self.assertEqual((r["img_w"], r["img_h"]), (1280, 960))
        self.assertFalse({"GPS", "ADDRESS", "gps", "address"} & set(r))
        self.assertNotIn("36.1", json.dumps(r, ensure_ascii=False))


class GroupSplitTest(unittest.TestCase):
    def _df(self, n_groups=40, per_group=10, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for g in range(n_groups):
            for k in range(per_group):
                rows.append({"group": f"B{g % 5}_202207{g:02d}", "image": f"img{g}_{k // 3}",
                             "verdict": rng.choice(["재활용", "일반쓰레기", "별도배출"], p=[.45, .45, .10])})
        return pd.DataFrame(rows)

    def test_groups_never_cross_splits_and_ratios_are_close(self):
        df = self._df()
        df["split"] = ds.group_split(df, seed=42)
        self.assertTrue((df.groupby("group")["split"].nunique() == 1).all())
        self.assertTrue((df.groupby("image")["split"].nunique() == 1).all())
        frac = df["split"].value_counts(normalize=True)
        self.assertAlmostEqual(frac["train"], 0.70, delta=0.08)
        self.assertAlmostEqual(frac["val"], 0.15, delta=0.06)
        self.assertAlmostEqual(frac["test"], 0.15, delta=0.06)

    def test_same_seed_same_split(self):
        df = self._df()
        self.assertTrue(ds.group_split(df, 7).equals(ds.group_split(df, 7)))

    def test_group_of_uses_method_and_day(self):
        self.assertEqual(ds.group_of("B2_20220721_000024.jpg"), "B2_20220721")
        self.assertEqual(ds.group_of("B2_20220721_000024.jpg", "image"), "B2_20220721_000024")


class CropAndDedupTest(unittest.TestCase):
    def test_crop_is_square_and_handles_borders(self):
        img = np.zeros((100, 200, 3), np.uint8)
        out = ds.crop_box(img, x=180, y=80, w=40, h=30)   # 박스가 이미지 밖으로 나감
        self.assertEqual(out.shape, (ds.CROP_SIZE, ds.CROP_SIZE, 3))

    def test_cross_split_duplicates_dropped_from_val_test(self):
        df = pd.DataFrame({"dhash": ["a", "a", "a", "b"], "split": ["test", "train", "val", "val"]})
        out, n = ds.drop_cross_split_duplicates(df)
        self.assertEqual(n, 2)
        self.assertEqual(sorted(out["split"]), ["train", "val"])

    def test_sampling_caps_each_code_and_drops_small_boxes(self):
        df = pd.DataFrame({"object_id": [f"o{i}" for i in range(30)], "code": ["c_1"] * 20 + ["c_9"] * 10,
                           "w": [10] * 5 + [100] * 25, "h": [100] * 30, "env": "실내형분류기"})
        out = ds.filter_and_sample(df, per_code=8, seed=1)
        self.assertEqual(out["code"].value_counts().to_dict(), {"c_1": 8, "c_9": 8})
        self.assertTrue((out["w"] >= ds.MIN_SIDE).all())


class MetricsTest(unittest.TestCase):
    C = ["재활용", "일반쓰레기", "별도배출"]

    def test_misguide_rate_counts_general_predicted_as_recycle(self):
        y = ["일반쓰레기", "일반쓰레기", "일반쓰레기", "일반쓰레기", "재활용"]
        p = ["재활용", "일반쓰레기", "별도배출", "일반쓰레기", "일반쓰레기"]
        self.assertAlmostEqual(ev.misguide_rate(y, p), 0.25)

    def test_battery_recall_and_contamination_f1(self):
        y = ["별도배출", "별도배출", "일반쓰레기", "재활용"]
        p = ["별도배출", "재활용", "일반쓰레기", "재활용"]
        items = ["건전지", "건전지", "플라스틱", "플라스틱"]
        m = ev.verdict_metrics(y, p, self.C, items)
        self.assertAlmostEqual(m["battery_recall"], 0.5)
        self.assertAlmostEqual(m["item_contam_f1"]["플라스틱"], 1.0)
        self.assertIsNone(m["item_contam_f1"]["건전지"])

    def test_ece_zero_when_confidence_matches_accuracy(self):
        proba = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        self.assertAlmostEqual(ev.ece_score(proba, ["재활용", "일반쓰레기"], self.C), 0.0)

    def test_pick_tau_returns_lowest_tau_meeting_target(self):
        curve = [{"tau": 0.5, "auto_misguide_rate": 0.10}, {"tau": 0.6, "auto_misguide_rate": 0.04},
                 {"tau": 0.7, "auto_misguide_rate": 0.02}]
        self.assertEqual(ev.pick_tau(curve, 0.05)["tau"], 0.6)
        self.assertIsNone(ev.pick_tau(curve, 0.01))


if __name__ == "__main__":
    unittest.main()
