"""src/preprocess/aihub_download.py 단위 테스트 (네트워크 없이 실행).

실행: python -m unittest discover -s tests -v
"""
import io
import os
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from src.preprocess import aihub_download as ad

CONFIG = {
    "dataset_key": "71385",
    "raw_subdir": "raw/aihub_71385",
    "presets": {"a": {"filekeys": [1, 2]}, "b": {"filekeys": [2, 3]}},
}


class ResolveFilekeysTest(unittest.TestCase):
    def test_presets_and_manual_keys_are_merged_without_duplicates(self):
        self.assertEqual(ad.resolve_filekeys(CONFIG, ["a", "b"], ["3,4", "5"]), [1, 2, 3, 4, 5])

    def test_unknown_preset_raises(self):
        with self.assertRaises(ad.AIHubError):
            ad.resolve_filekeys(CONFIG, ["nope"], [])

    def test_real_config_presets_have_no_duplicates(self):
        cfg = ad.load_config()
        for name, p in cfg["presets"].items():
            self.assertEqual(len(p["filekeys"]), len(set(p["filekeys"])), name)


class MergePartsTest(unittest.TestCase):
    def test_parts_are_joined_in_numeric_offset_order(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "sub"
            root.mkdir()
            # 문자열 순서(part0, part10, part5)와 숫자 순서(0, 5, 10)가 다른 경우
            (root / "A.zip.part0").write_bytes(b"AAAAA")
            (root / "A.zip.part10").write_bytes(b"CC")
            (root / "A.zip.part5").write_bytes(b"BBBBB")
            merged = ad.merge_parts(Path(d))
            self.assertEqual(merged, [root / "A.zip"])
            self.assertEqual((root / "A.zip").read_bytes(), b"AAAAABBBBBCC")
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["A.zip"])


class SafeExtractTarTest(unittest.TestCase):
    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            tar_path = Path(d) / "evil.tar"
            with tarfile.open(tar_path, "w") as tf:
                data = b"x"
                info = tarfile.TarInfo("../evil.txt")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            with self.assertRaises(ad.AIHubError):
                ad.safe_extract_tar(tar_path, Path(d) / "out")
            self.assertFalse((Path(d) / "evil.txt").exists())


class ZipNameTest(unittest.TestCase):
    def test_cp949_name_without_utf8_flag_is_decoded(self):
        name = "실내형분류기/라벨.json"
        info = zipfile.ZipInfo(name.encode("cp949").decode("cp437"))
        info.flag_bits = 0
        self.assertEqual(ad._zip_member_name(info), name)

    def test_leading_slash_is_stripped(self):
        # 실제 AI Hub zip (TL_실내형분류기_B2.zip)의 항목 이름 형식
        info = zipfile.ZipInfo("/B2_20220721_000024.json")
        info.flag_bits = 0x800
        self.assertEqual(ad._zip_member_name(info), "B2_20220721_000024.json")

    def test_utf8_flagged_name_is_kept(self):
        info = zipfile.ZipInfo("종이/a.jpg")
        info.flag_bits = 0x800
        self.assertEqual(ad._zip_member_name(info), "종이/a.jpg")


class ExtractZipsTest(unittest.TestCase):
    def test_matching_zip_is_extracted_and_others_skipped(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"ML_TEAMWORK_DATA": d}):
            raw = Path(d) / "raw" / "aihub_71385" / "Training"
            raw.mkdir(parents=True)
            for zname in ["TL_실내형분류기_B2.zip", "TL_재활용선별장_A2.zip"]:
                with zipfile.ZipFile(raw / zname, "w") as zf:
                    zf.writestr("label/1.json", "{}")
            dest = Path(d) / "work"
            out = ad.extract_zips(CONFIG, "*실내형분류기*", dest)
            self.assertEqual(out, [dest / "TL_실내형분류기_B2"])
            self.assertTrue((dest / "TL_실내형분류기_B2" / "label" / "1.json").exists())
            self.assertFalse((dest / "TL_재활용선별장_A2").exists())

    def test_parent_dir_entry_is_rejected(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"ML_TEAMWORK_DATA": d}):
            raw = Path(d) / "raw" / "aihub_71385"
            raw.mkdir(parents=True)
            with zipfile.ZipFile(raw / "evil.zip", "w") as zf:
                zf.writestr("/../../escape.txt", "x")
            with self.assertRaises(ad.AIHubError):
                ad.extract_zips(CONFIG, "*.zip", Path(d) / "work")
            self.assertFalse((Path(d) / "escape.txt").exists())


class ResumeTest(unittest.TestCase):
    def test_filekeys_in_manifest_with_existing_file_are_skipped(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
                os.environ, {"ML_TEAMWORK_DATA": d, "AIHUB_APIKEY": "test-key"}):
            raw = Path(d) / "raw" / "aihub_71385"
            (raw / "T").mkdir(parents=True)
            (raw / "T" / "a.zip").write_bytes(b"x")
            ad._append_manifest(raw, [
                {"downloaded_at": "", "filekey": 1, "path": "T/a.zip", "size_bytes": 1, "sha256": ""},
                {"downloaded_at": "", "filekey": 2, "path": "T/gone.zip", "size_bytes": 1, "sha256": ""}])
            self.assertEqual(ad.downloaded_filekeys(raw), {1})
            with mock.patch.object(ad, "_download_tar") as dl:
                ad.download(CONFIG, [1], work_dir=Path(d) / "w")
                dl.assert_not_called()


class RetryTest(unittest.TestCase):
    def test_dropped_connection_is_retried_from_scratch(self):
        import http.client
        calls = []

        def flaky(url, key, dest):
            calls.append(1)
            if len(calls) < 3:
                raise http.client.IncompleteRead(b"x")
            dest.write_bytes(b"done")

        with tempfile.TemporaryDirectory() as d, mock.patch.object(ad, "_download_once", flaky):
            dest = Path(d) / "download.tar"
            ad._download_tar("71385", [1], "k", dest, retries=5, wait=0)
            self.assertEqual(len(calls), 3)
            self.assertEqual(dest.read_bytes(), b"done")

    def test_gives_up_after_retries_and_auth_errors_are_not_retried(self):
        import http.client
        with mock.patch.object(ad, "_download_once", side_effect=http.client.IncompleteRead(b"")) as m:
            with self.assertRaises(ad.AIHubError):
                ad._download_tar("71385", [1], "k", Path("unused"), retries=3, wait=0)
            self.assertEqual(m.call_count, 3)
        with mock.patch.object(ad, "_download_once", side_effect=ad.AIHubError("인증실패")) as m:
            with self.assertRaises(ad.AIHubError):
                ad._download_tar("71385", [1], "k", Path("unused"), retries=3, wait=0)
            self.assertEqual(m.call_count, 1)


class ApiKeyTest(unittest.TestCase):
    def test_key_is_extracted_from_surrounding_text(self):
        key = "ABCDEF01-2345-6789-ABCD-EF0123456789"
        for raw in (key, f"  {key}\n", f"키: {key}", f"ㅇㅇㅇ{key}"):
            with mock.patch.dict(os.environ, {"AIHUB_APIKEY": raw}):
                self.assertEqual(ad.get_api_key(), key)

    def test_non_ascii_key_gives_clear_error(self):
        with mock.patch.dict(os.environ, {"AIHUB_APIKEY": "ㅇㅇㅇ1234"}):
            with self.assertRaisesRegex(ad.AIHubError, "한글"):
                ad.get_api_key()

    def test_missing_key_raises(self):
        with mock.patch.dict(os.environ, {"AIHUB_APIKEY": ""}):
            with self.assertRaises(ad.AIHubError):
                ad.get_api_key()


if __name__ == "__main__":
    unittest.main()
