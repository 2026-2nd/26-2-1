"""Colab 노트북에서 Google Drive·GitHub를 연결하고, 작업을 GitHub 브랜치에 올리는 도우미.

노트북 첫 셀   colab.setup("model/knn", new_branch=True)   # Drive 연결 + 브랜치 준비
노트북 마지막 셀 colab.push("knn: k 값 비교 추가")           # 노트북·코드 변경을 브랜치에 올림
그 밖에         colab.branches()                            # GitHub에 있는 브랜치 목록
               colab.check_github()                        # 토큰이 제대로 설정됐는지 확인

규칙 (docs/COLLABORATION.md):
  - main에는 직접 올리지 않는다. push()는 main에서 멈춘다.
  - 브랜치 이름은 `<종류>/<이름>` 이다. 새 모델은 반드시 새 `model/<모델명>` 브랜치.
  - 데이터·모델 파일은 Drive에 두고 Git에 올리지 않는다. 큰 파일(5MB 초과)은 push()가 거부한다.

코드는 런타임의 `/content/26-2-1`에 GitHub에서 새로 받은 복사본이다. 런타임이 끝나면 사라지므로
고친 내용은 push()로 올려야 남는다. Drive(`MyDrive/ML_TeamWork`)에는 데이터·모델·출력물만 둔다.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

from . import paths

REPO_SLUG = "2026-2nd/26-2-1"
REPO_URL = f"https://github.com/{REPO_SLUG}.git"
MAIN = "main"
MAX_FILE_MB = 5

# 브랜치 종류 → 설명, 이 종류의 노트북을 저장할 폴더
BRANCH_KINDS = {
    "model": ("모델 하나 (예: model/knn). 모델마다 브랜치 하나", "notebooks/2_models"),
    "preprocess": ("전처리 작업 (예: preprocess/crop)", "notebooks/1_preprocess"),
    "setup": ("Colab·Drive·Git 연결, 환경", None),
    "docs": ("문서만 수정", None),
    "fix": ("이미 있는 코드의 오류 수정", None),
}
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

_state: dict = {"notebook": None}


class ColabError(RuntimeError):
    """사용자가 고쳐야 하는 문제. 메시지에 해결 방법을 적는다."""


# ── 판단 함수 (Colab 없이도 테스트 가능) ───────────────────────────────
def in_colab() -> bool:
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


def branch_problem(name: str) -> str | None:
    """작업 브랜치로 쓸 수 없는 이름이면 이유를, 괜찮으면 None을 돌려준다."""
    if name == MAIN:
        return "main에는 직접 올리지 않습니다. 작업 브랜치(예: model/knn)를 쓰세요."
    kind, sep, rest = name.partition("/")
    if not sep or kind not in BRANCH_KINDS:
        kinds = ", ".join(f"{k}/" for k in BRANCH_KINDS)
        return f"브랜치 이름은 '<종류>/<이름>' 형식이어야 합니다. 종류: {kinds} (예: model/knn)"
    if not _NAME_RE.match(rest):
        return "'/' 뒤에는 영어 소문자·숫자·'-'·'_'·'.'만 씁니다 (예: model/knn, preprocess/crop-224)."
    return None


def default_notebook(branch: str) -> str | None:
    """model/knn → notebooks/2_models/knn.ipynb. 노트북 폴더가 없는 종류는 None."""
    kind, _, rest = branch.partition("/")
    folder = BRANCH_KINDS.get(kind, (None, None))[1]
    return f"{folder}/{rest}.ipynb" if folder and rest else None


def notebook_problem(path: str) -> str | None:
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or p.parts[:1] != ("notebooks",) or p.suffix != ".ipynb":
        return "노트북은 'notebooks/.../이름.ipynb' 형태의 저장소 안 경로로 지정합니다."
    if p.name.startswith("_template"):
        return "템플릿 파일은 덮어쓰지 않습니다. 새 이름을 지정하세요 (예: notebooks/2_models/knn.ipynb)."
    return None


def _scrub(text: str, token: str | None) -> str:
    if not token:
        return text
    for secret in (token, _basic(token)):
        text = text.replace(secret, "***")
    return text


def _basic(token: str) -> str:
    return base64.b64encode(f"x-access-token:{token}".encode()).decode()


# ── git 실행 ──────────────────────────────────────────────────────────
def _git(*args: str, check: bool = True, token: str | None = None) -> tuple[int, str]:
    cmd = ["git"]
    if token:   # 토큰을 URL이나 설정 파일에 남기지 않고 이번 명령에만 쓴다
        cmd += ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {_basic(token)}"]
    cmd += list(args)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    r = subprocess.run(cmd, cwd=paths.REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    out = _scrub((r.stdout + r.stderr).strip(), token)
    if check and r.returncode != 0:
        raise ColabError(f"git {args[0]} 실패:\n{out}")
    return r.returncode, out


def _ref_exists(ref: str) -> bool:
    return _git("rev-parse", "--verify", "--quiet", ref, check=False)[0] == 0


def current_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")[1]


def _dirty_files() -> list[str]:
    out = _git("status", "--porcelain")[1]
    return [line[3:] for line in out.splitlines() if line.strip()]


# ── 첫 셀: 준비 ───────────────────────────────────────────────────────
def setup(branch: str = MAIN, new_branch: bool = False, notebook: str | None = None,
          drive_dir: str | None = None) -> None:
    """Drive를 연결하고 GitHub 브랜치를 준비한다.

    branch      작업할 브랜치. 읽기만 할 때는 "main".
    new_branch  GitHub에 아직 없는 브랜치를 main에서 새로 만들 때 True.
    notebook    push()가 이 노트북을 저장할 위치. 생략하면 브랜치 이름으로 정한다
                (model/knn → notebooks/2_models/knn.ipynb).
    drive_dir   Drive의 프로젝트 폴더. 기본 /content/drive/MyDrive/ML_TeamWork
    """
    if not in_colab():
        print("Colab이 아닙니다. 로컬에서는 git 명령으로 브랜치를 관리합니다 (docs/COLLABORATION.md).")
        print(paths.describe())
        return

    print("① Google Drive 연결")
    from google.colab import drive
    drive.mount("/content/drive")
    project = Path(drive_dir) if drive_dir else paths.COLAB_DRIVE_DIR
    if not project.is_dir():
        raise ColabError(
            f"Drive에 {project} 폴더가 없습니다.\n"
            "  공유받은 폴더라면: drive.google.com → '공유 문서함' → ML_TeamWork 폴더 오른쪽 클릭 →\n"
            "  '정리' → '바로가기 추가' → '내 드라이브'를 고른 뒤 이 셀을 다시 실행하세요.")
    os.environ["ML_TEAMWORK_DIR"] = str(project)
    print(paths.describe())

    print("\n② GitHub 브랜치 준비")
    _prepare_branch(branch, new_branch)
    _forget_src_modules()

    nb = notebook or (default_notebook(branch) if branch != MAIN else None)
    if nb and (problem := notebook_problem(nb)):
        raise ColabError(problem)
    _state["notebook"] = nb

    sha = _git("log", "-1", "--format=%h %s")[1]
    print(f"  브랜치: {current_branch()}  (최신 커밋: {sha})")
    if branch == MAIN:
        print("  main은 읽기 전용입니다. 고친 내용을 올리려면 작업 브랜치로 바꾸세요.")
    elif nb:
        print(f"  마지막 셀의 colab.push()가 이 노트북을 {nb} 로 저장합니다.")
    print("\n준비 완료. 아래 셀을 위에서부터 차례로 실행하세요.")


def _prepare_branch(branch: str, new_branch: bool) -> None:
    _git("fetch", "--prune", "origin")
    current = current_branch()
    on_remote = _ref_exists(f"refs/remotes/origin/{branch}")
    on_local = _ref_exists(f"refs/heads/{branch}")

    if branch != MAIN and (problem := branch_problem(branch)):
        raise ColabError(problem)
    if new_branch and on_remote:
        raise ColabError(f"'{branch}'는 이미 GitHub에 있습니다. NEW_BRANCH = False 로 바꾸고 다시 실행하세요.")
    if not new_branch and not on_remote and not on_local:
        similar = [b for b in remote_branches() if b.split("/")[0] == branch.split("/")[0]]
        hint = f"\n  같은 종류의 브랜치: {', '.join(similar)}" if similar else ""
        raise ColabError(f"GitHub에 '{branch}' 브랜치가 없습니다. 이름을 확인하세요.{hint}\n"
                         "  새로 만들 거라면 NEW_BRANCH = True 로 바꾸고 다시 실행하세요.")

    if current != branch:
        if dirty := _dirty_files():
            raise ColabError(f"'{current}'에서 고치고 아직 올리지 않은 파일이 있습니다: {', '.join(dirty[:5])}\n"
                             "  colab.push()로 먼저 올리거나, 버려도 되면 '런타임 → 런타임 연결 해제 및 삭제' 후 다시 시작하세요.")
        if on_local:
            _git("switch", branch)
        elif on_remote:
            _git("switch", "-c", branch, "--track", f"origin/{branch}")
        else:
            _git("switch", "-c", branch, "--no-track", f"origin/{MAIN}")
            print(f"  main에서 새 브랜치 '{branch}'를 만들었습니다. 마지막 셀에서 push하면 GitHub에도 생깁니다.")
            return

    if on_remote:
        code, out = _git("merge", "--ff-only", f"origin/{branch}", check=False)
        if code != 0:
            raise ColabError("GitHub의 최신 코드와 이 런타임의 코드를 자동으로 합칠 수 없습니다.\n"
                             "  이 런타임에서 만든 커밋을 아직 올리지 않았다면 colab.push()를 먼저 실행하세요.\n" + out)


def _forget_src_modules() -> None:
    """브랜치를 바꾼 뒤 예전 브랜치의 src 모듈을 다시 불러오도록 캐시를 지운다."""
    for name in list(sys.modules):
        if (name == "src" or name.startswith("src.")) and not name.startswith("src.common"):
            del sys.modules[name]


# ── 브랜치 목록 ───────────────────────────────────────────────────────
def remote_branches() -> list[str]:
    out = _git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin")[1]
    names = [line.removeprefix("origin/") for line in out.splitlines()]
    return sorted(n for n in names if n and n not in ("HEAD", "origin"))


def branches() -> None:
    """GitHub의 브랜치를 종류별로 보여 준다. 새 브랜치를 만들기 전에 겹치는 이름이 없는지 확인한다."""
    _git("fetch", "--prune", "origin")
    groups: dict[str, list[str]] = {}
    for b in remote_branches():
        groups.setdefault(b.split("/")[0] if "/" in b else b, []).append(b)
    for kind, names in sorted(groups.items()):
        desc = BRANCH_KINDS.get(kind, ("",))[0]
        print(f"[{kind}] {desc}")
        for n in names:
            print(f"   {n}")


# ── GitHub 토큰 ───────────────────────────────────────────────────────
def _token() -> str:
    token = ""
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN") or ""
    except Exception:   # Secrets에 없거나 이 노트북에 접근을 허용하지 않음
        pass
    if not token:
        from getpass import getpass
        print("Colab 🔑 Secrets에 GITHUB_TOKEN이 없거나 이 노트북에서 사용을 허용하지 않았습니다.")
        token = getpass("GitHub 토큰을 붙여 넣고 Enter (화면에 보이지 않고 노트북에 저장되지 않음): ")
    token = token.strip()
    if not token:
        raise ColabError("GitHub 토큰이 비어 있습니다. docs/SETUP_GUIDE.md의 토큰 만들기를 따라 하세요.")
    return token


def _api(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _github_identity(token: str) -> tuple[str, str]:
    """토큰을 확인하고 (GitHub 아이디, 커밋에 쓸 noreply 이메일)을 돌려준다."""
    try:
        user = _api("https://api.github.com/user", token)
    except urllib.error.HTTPError as e:
        raise ColabError(f"GitHub 토큰이 잘못됐거나 만료됐습니다 (HTTP {e.code}). 새로 만들어 Secrets를 바꾸세요.") from None
    try:
        repo = _api(f"https://api.github.com/repos/{REPO_SLUG}", token)
    except urllib.error.HTTPError as e:
        raise ColabError(f"이 토큰으로 {REPO_SLUG} 저장소를 볼 수 없습니다 (HTTP {e.code}).\n"
                         "  토큰의 Resource owner가 2026-2nd인지, 저장소 26-2-1을 골랐는지 확인하세요.") from None
    if repo.get("permissions") is not None and not repo["permissions"].get("push"):
        raise ColabError(f"{user['login']} 계정에 {REPO_SLUG} 쓰기 권한이 없습니다.\n"
                         "  저장소 초대를 수락했는지, 토큰 권한 Contents가 'Read and write'인지 확인하세요.")
    return user["login"], f"{user['id']}+{user['login']}@users.noreply.github.com"


def check_github() -> None:
    """토큰과 저장소 쓰기 권한을 확인한다 (아무것도 올리지 않음)."""
    login, _ = _github_identity(_token())
    print(f"✅ GitHub 연결 확인: {login} 계정으로 {REPO_SLUG}에 올릴 수 있습니다.")


# ── 마지막 셀: 올리기 ─────────────────────────────────────────────────
def save_notebook(path: str, keep_outputs: bool = False) -> Path:
    """지금 열려 있는 Colab 노트북 내용을 저장소 안 `path`에 저장한다."""
    if problem := notebook_problem(path):
        raise ColabError(problem)
    try:
        from google.colab import _message
        nb = _message.blocking_request("get_ipynb", timeout_sec=60)["ipynb"]
    except Exception as e:
        raise ColabError(f"노트북 내용을 읽지 못했습니다 ({type(e).__name__}).\n"
                         "  '파일 → GitHub에 사본 저장'으로 직접 저장하거나, notebook=None으로 코드만 올리세요.") from None
    if not keep_outputs:   # 출력은 지워서 충돌과 데이터 노출을 줄인다
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"], cell["execution_count"] = [], None
    dest = paths.REPO_ROOT / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return dest


def push(message: str, notebook: str | None = None, keep_outputs: bool = False) -> None:
    """이 노트북과 고친 코드를 지금 브랜치에 커밋하고 GitHub에 올린다.

    message       무엇을 왜 바꿨는지 (예: "knn: k=1~15 비교 추가")
    notebook      저장할 위치. 생략하면 setup()에서 정한 위치
    keep_outputs  True면 셀 출력도 저장 (기본은 지움)
    """
    if not in_colab():
        raise ColabError("Colab이 아닙니다. 로컬에서는 git add / commit / push를 직접 실행합니다.")
    if not message.strip():
        raise ColabError("커밋 메시지를 적어 주세요. 예: colab.push(\"knn: k 값 비교 추가\")")
    branch = current_branch()
    if problem := branch_problem(branch):
        raise ColabError(problem)

    nb = notebook or _state["notebook"]
    if nb:
        print(f"노트북 저장: {save_notebook(nb, keep_outputs).relative_to(paths.REPO_ROOT).as_posix()}")
    else:
        print("저장할 노트북 위치가 없어 코드 변경만 올립니다.")

    _git("add", "-A")
    staged = [f for f in _git("diff", "--cached", "--name-only")[1].splitlines() if f]
    big = [f for f in staged if (paths.REPO_ROOT / f).is_file()
           and (paths.REPO_ROOT / f).stat().st_size > MAX_FILE_MB * 2**20]
    if big:
        _git("reset", "-q", "--", *big)
        raise ColabError(f"{MAX_FILE_MB}MB보다 큰 파일은 Git에 올리지 않습니다: {', '.join(big)}\n"
                         "  데이터·모델 파일은 paths.data_dir() / paths.models_dir() (Drive)에 저장하세요.")

    on_remote = _ref_exists(f"refs/remotes/origin/{branch}")
    ahead = _git("rev-list", "--count", f"origin/{branch if on_remote else MAIN}..HEAD")[1]
    if not staged and ahead == "0":
        print("올릴 변경이 없습니다.")
        return
    if staged:
        print("올릴 파일:\n  " + "\n  ".join(staged))

    token = _token()
    login, email = _github_identity(token)
    _git("config", "user.name", login)
    _git("config", "user.email", email)
    if staged:
        _git("commit", "-q", "-m", message)

    if on_remote:   # 같은 브랜치에서 다른 사람이 먼저 올린 내용을 먼저 합친다
        _git("fetch", "origin", branch, token=token)
        code, out = _git("rebase", f"origin/{branch}", check=False)
        if code != 0:
            _git("rebase", "--abort", check=False)
            raise ColabError("같은 브랜치에서 다른 사람이 같은 부분을 고쳐 자동으로 합칠 수 없습니다.\n"
                             "  내 커밋은 이 런타임에 남아 있습니다. 런타임을 끄지 말고 팀원과 상의하세요.\n" + out)
    _git("push", "-u", "origin", f"HEAD:refs/heads/{branch}", token=token)

    print(f"\n✅ {login} → {branch} 에 올렸습니다.")
    print(f"  GitHub에서 보기: https://github.com/{REPO_SLUG}/tree/{branch}")
    print(f"  main에 합치려면 PR: https://github.com/{REPO_SLUG}/compare/{MAIN}...{branch}?expand=1")
    if nb:
        print("  다음에 이어서 하려면 Colab → 파일 → 노트북 열기 → GitHub 탭에서 "
              f"{REPO_SLUG}, 브랜치 {branch}, {nb} 를 여세요.")
