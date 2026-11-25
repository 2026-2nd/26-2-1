"""src/common (paths, colab) 단위 테스트. Colab·GitHub 없이 실행한다.

실행: python -m unittest discover -s tests -v

git 동작은 임시 폴더에 만든 가짜 GitHub(bare 저장소)와 두 개의 복사본으로 확인한다.
"""
import datetime as dt
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.common import colab, paths


class BranchRuleTest(unittest.TestCase):
    def test_good_names(self):
        for name in ["model/knn", "preprocess/crop-224", "docs/colab-guide", "fix/path_bug", "setup/drive"]:
            self.assertIsNone(colab.branch_problem(name), name)

    def test_bad_names(self):
        for name in ["main", "knn", "models/knn", "model/", "model/KNN", "model/knn/v2", "model/케이엔엔"]:
            self.assertIsNotNone(colab.branch_problem(name), name)

    def test_default_notebook_follows_branch_kind(self):
        self.assertEqual(colab.default_notebook("model/knn"), "notebooks/2_models/knn.ipynb")
        self.assertEqual(colab.default_notebook("preprocess/crop"), "notebooks/1_preprocess/crop.ipynb")
        self.assertIsNone(colab.default_notebook("docs/readme"))

    def test_notebook_path_rules(self):
        self.assertIsNone(colab.notebook_problem("notebooks/2_models/knn.ipynb"))
        for bad in ["knn.ipynb", "notebooks/../x.ipynb", "/content/x.ipynb", "notebooks/2_models/knn.py",
                    "notebooks/2_models/_template_model.ipynb"]:
            self.assertIsNotNone(colab.notebook_problem(bad), bad)

    def test_token_is_scrubbed(self):
        text = f"fatal: ghp_SECRET and {colab._basic('ghp_SECRET')}"
        self.assertNotIn("SECRET", colab._scrub(text, "ghp_SECRET"))


class PathsTest(unittest.TestCase):
    def test_env_overrides(self):
        with mock.patch.dict(os.environ, {"ML_TEAMWORK_DIR": "/p"}, clear=False):
            os.environ.pop("ML_TEAMWORK_DATA", None)
            self.assertEqual(paths.data_dir(), Path("/p/data"))
            self.assertEqual(paths.models_dir(), Path("/p/models"))
            with mock.patch.dict(os.environ, {"ML_TEAMWORK_DATA": "/d"}):
                self.assertEqual(paths.data_dir(), Path("/d"))

    def test_run_dir_name(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(paths, "REPO_ROOT", Path(d)):
            p = paths.run_dir("knn k=5!", now=dt.datetime(2026, 11, 25, 9, 30, 0))
            self.assertEqual(p.relative_to(d).as_posix(), "reports/experiments/20261125-093000_knn-k-5")
            self.assertTrue(p.is_dir())


def _run(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8")


def _out(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


class GitFlowTest(unittest.TestCase):
    """가짜 GitHub(origin)를 두고 setup의 브랜치 준비와 push의 git 단계를 확인한다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.origin = root / "origin.git"
        _run(root, "init", "-q", "--bare", "-b", "main", str(self.origin))
        seed = root / "seed"
        _run(root, "clone", "-q", str(self.origin), str(seed))
        for c in (["config", "user.name", "t"], ["config", "user.email", "t@example.com"]):
            _run(seed, *c)
        (seed / "README.md").write_text("hi\n", encoding="utf-8")
        _run(seed, "add", "README.md")
        _run(seed, "commit", "-q", "-m", "init")
        _run(seed, "push", "-q", "origin", "main")
        self.seed = seed
        self.repo = root / "colab"
        _run(root, "clone", "-q", str(self.origin), str(self.repo))
        patches = [
            mock.patch.object(paths, "REPO_ROOT", self.repo),
            mock.patch.object(colab, "in_colab", return_value=True),
            mock.patch.object(colab, "_token", return_value="tok"),
            mock.patch.object(colab, "_github_identity", return_value=("tester", "1+tester@users.noreply.github.com")),
            mock.patch.object(colab, "_state", {"notebook": None}),
            mock.patch("builtins.print"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_new_branch_then_push(self):
        colab._prepare_branch("model/knn", new_branch=True)
        self.assertEqual(colab.current_branch(), "model/knn")
        (self.repo / "knn.py").write_text("k = 5\n", encoding="utf-8")
        colab.push("knn 추가")
        self.assertEqual(_out(self.origin, "log", "-1", "--format=%s %an", "model/knn"), "knn 추가 tester")
        self.assertEqual(_out(self.origin, "log", "-1", "--format=%s", "main"), "init")   # main은 그대로

    def test_push_refuses_main(self):
        (self.repo / "x.py").write_text("x\n", encoding="utf-8")
        with self.assertRaises(colab.ColabError):
            colab.push("main에 올리기")

    def test_existing_branch_requires_flag_and_missing_branch_is_explained(self):
        with self.assertRaises(colab.ColabError):
            colab._prepare_branch("model/nope", new_branch=False)
        colab._prepare_branch("model/knn", new_branch=True)
        (self.repo / "a.py").write_text("a\n", encoding="utf-8")
        colab.push("a")
        with self.assertRaises(colab.ColabError):   # 이미 GitHub에 있으면 NEW_BRANCH=True 금지
            colab._prepare_branch("model/knn", new_branch=True)

    def test_switch_is_blocked_with_unpushed_changes(self):
        colab._prepare_branch("model/knn", new_branch=True)
        (self.repo / "wip.py").write_text("wip\n", encoding="utf-8")
        with self.assertRaises(colab.ColabError):
            colab._prepare_branch("model/svm", new_branch=True)
        self.assertTrue((self.repo / "wip.py").exists())   # 작업이 지워지지 않음

    def test_push_merges_teammate_commit_on_same_branch(self):
        colab._prepare_branch("model/knn", new_branch=True)
        (self.repo / "a.py").write_text("a\n", encoding="utf-8")
        colab.push("a")
        _run(self.seed, "fetch", "-q", "origin")
        _run(self.seed, "switch", "-q", "model/knn")
        (self.seed / "b.py").write_text("b\n", encoding="utf-8")
        _run(self.seed, "add", "b.py")
        _run(self.seed, "commit", "-q", "-m", "teammate b")
        _run(self.seed, "push", "-q", "origin", "model/knn")
        (self.repo / "c.py").write_text("c\n", encoding="utf-8")
        colab.push("c")
        log = _out(self.origin, "log", "--format=%s", "model/knn").splitlines()
        self.assertEqual(log[:3], ["c", "teammate b", "a"])

    def test_setup_switches_to_existing_branch_and_pulls(self):
        colab._prepare_branch("model/knn", new_branch=True)
        (self.repo / "a.py").write_text("a\n", encoding="utf-8")
        colab.push("a")
        _run(self.repo, "switch", "-q", "main")
        colab._prepare_branch("model/knn", new_branch=False)
        self.assertTrue((self.repo / "a.py").exists())

    def test_big_file_is_refused(self):
        colab._prepare_branch("model/knn", new_branch=True)
        (self.repo / "weights.bin").write_bytes(b"0" * (colab.MAX_FILE_MB * 2**20 + 1))
        with self.assertRaises(colab.ColabError):
            colab.push("모델 파일")
        self.assertEqual(_out(self.repo, "diff", "--cached", "--name-only"), "")


if __name__ == "__main__":
    unittest.main()


class GitignoreTest(unittest.TestCase):
    """팀원이 만드는 코드는 Git에 들어가고, 데이터·모델·비밀은 빠지는지 실제 .gitignore로 확인한다."""

    @classmethod
    def setUpClass(cls):   # PC마다 다른 .git/info/exclude의 영향을 받지 않도록 .gitignore만 복사한 빈 저장소에서 검사
        cls.tmp = tempfile.TemporaryDirectory()
        _run(cls.tmp.name, "init", "-q")
        (Path(cls.tmp.name) / ".gitignore").write_bytes((paths.REPO_ROOT / ".gitignore").read_bytes())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _ignored(self, rel):
        return subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=self.tmp.name).returncode == 0

    def test_code_is_tracked(self):
        for rel in ["src/models/knn.py", "src/preprocess/crop.py", "notebooks/2_models/knn.ipynb",
                    "reports/experiments/20261125-093000_knn/record.json"]:
            self.assertFalse(self._ignored(rel), rel)

    def test_data_models_secrets_are_ignored(self):
        for rel in ["data/raw/a.zip", "models/knn/model.pkl", "outputs/x.png", ".env", "src/desktop.ini",
                    "weights.zip"]:
            self.assertTrue(self._ignored(rel), rel)
