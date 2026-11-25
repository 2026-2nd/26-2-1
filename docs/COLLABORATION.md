# 협업 규칙 — 브랜치 · 커밋 · PR · AI 에이전트

Git으로 팀 프로젝트를 처음 하는 팀원을 위한 규칙이다. Colab·Drive·GitHub·내 PC 연결은 [연결 튜토리얼](SETUP_GUIDE.md)을 먼저 따라 한다.

대상 저장소는 `2026-2nd/26-2-1`(Public), 기본 브랜치는 `main`이다. 이 문서는 GitHub 설정(브랜치 보호·필수 리뷰 등)을 자동으로 바꾸지 않는다. 담당자는 아직 정하지 않았다.

## 1. 한눈에 보는 규칙 (2026-11-25 확정)

1. **`main`은 모두가 쓰는 완성본이다. 직접 고치지 않고 PR로만 합친다.**
2. **브랜치는 세 종류뿐이다.**

   | 브랜치 | 하는 일 | 새로 만드나 |
   | --- | --- | --- |
   | `common` | 문서 추가·수정, 공통 코드(`src/common/`), 설정, 환경, 오류 수정 | ❌ 이미 있음. 모두 여기서 |
   | `preprocess` | 모든 전처리 작업 | ❌ 이미 있음. 모두 여기서 |
   | `model/<모델명>` | 모델 하나 (예: `model/knn`) | ✅ **새 모델을 추가할 때만** |

3. **기존 모델을 고칠 때는 그 모델 이름의 브랜치에서 작업하고, 그 브랜치로 PR을 연다.** KNN 수정은 언제나 `model/knn`.
4. **이 세 종류 말고는 브랜치를 만들지 않는다.** `docs/...`, `fix/...`, `test`, 개인 이름 브랜치 모두 만들지 않는다.
5. **브랜치는 병합 후에도 지우지 않는다.** 계속 같은 브랜치에서 작업한다.
6. **데이터·모델 파일·토큰은 Git에 올리지 않는다.** 데이터·모델은 Drive, 토큰은 Secrets.

Colab 도우미(`colab.setup`, `colab.push`)는 이 규칙을 지키지 않으면 멈추고 이유를 알려 준다.

## 2. 브랜치 규칙 자세히

### 2-1. 어느 브랜치에서 일하나

| 하려는 일 | 브랜치 |
| --- | --- |
| README·docs 문서를 고치거나 추가 | `common` |
| Colab 연결 코드·경로·설정·requirements 수정 | `common` |
| `main`에 들어간 코드의 오류 수정 (모델 코드 제외) | `common` |
| 데이터 정리·crop·특징 만들기 등 전처리 | `preprocess` |
| 새 모델 KNN 추가 | `model/knn` **새로 만들기** |
| KNN의 k 값 바꾸기, KNN 오류 수정 | `model/knn` (이미 있음) |
| SVM과 KNN 비교 그래프 | 결과 정리용이면 `common`, 특정 모델의 실험이면 그 모델 브랜치 |

### 2-2. 모델 브랜치가 다룰 수 있는 파일

`model/knn` 브랜치에서 고치는 파일은 다음뿐이다.

```text
notebooks/2_models/knn*.ipynb
src/models/knn.py           (필요하면 src/models/knn/ 폴더)
reports/experiments/*_knn/  (실험 기록)
```

다른 모델 파일, 전처리 코드, 공통 코드, 문서를 고쳐야 하면 그 일은 `common`이나 `preprocess`에서 따로 한다. 그래야 PR 하나가 한 가지 일만 담고, 모델끼리 충돌하지 않는다.

### 2-3. 공용 브랜치(`common`, `preprocess`)에서 함께 일하기

여러 사람이 같은 브랜치에 올리므로:
- **작업 전에 최신 내용을 받는다.** Colab은 첫 셀이 자동으로 받는다. 로컬은 `git pull`.
- **각자 자기 파일에서 작업한다.** 전처리는 `notebooks/1_preprocess/<작업명>.ipynb`, `src/preprocess/<작업명>.py`처럼 작업 이름으로 나눈다. 남의 파일은 고치기 전에 먼저 이야기한다.
- **자주, 작게 올린다.** 오래 쌓아 두면 충돌이 커진다.
- `colab.push()`는 다른 사람이 먼저 올린 커밋을 자동으로 합친 뒤 올린다. 같은 파일의 같은 부분을 고쳐 합칠 수 없을 때만 멈춘다 (6절).

### 2-4. 흐름

```text
main ─────●───────────────●──────────────────●──────▶
          ↑ PR 병합        ↑ PR 병합          ↑ PR 병합
common ───┴──●──●─────────┴──●──────────────┘        (문서·공통: 계속 여기서)
preprocess ──●──●──●──────────┘                       (전처리: 계속 여기서)
model/knn ────●──●──(PR)──●──●──(PR)──▶               (KNN: 계속 여기서)
model/svm ──────────●──●──(PR)──▶                     (SVM: 새 모델이라 새로 만듦)
```

- 일정 단위(예: 기능 하나, 실험 하나)가 끝나면 그 브랜치에서 `main`으로 PR을 연다.
- 병합 후에도 같은 브랜치에서 계속 작업하고, 다음 단위가 끝나면 다시 PR을 연다.
- 다른 브랜치가 `main`에 합쳐져 내 브랜치가 뒤처지면 PR 화면의 **Update branch** 버튼으로 `main`의 변경을 받는다. 로컬에서는 `git merge origin/main`.

### 2-5. 브랜치 목록 확인

새 모델 브랜치를 만들기 전에 같은 모델이 이미 있는지 확인한다.
- Colab: [연결 확인 노트북](../notebooks/0_setup/check_connection.ipynb)의 `colab.branches()` (규칙 밖 브랜치도 표시)
- GitHub: 저장소 → **Branches**
- 로컬: `git fetch --prune origin` 후 `git branch -r`

## 3. 작업 순서

### 3-1. Colab에서 (권장)

[연결 튜토리얼 5절](SETUP_GUIDE.md#5-colab에서-일하는-법-매번)대로 한다. 첫 셀 설정만 정리하면 이렇다.

| 상황 | 여는 노트북 | 첫 셀 |
| --- | --- | --- |
| 전처리 | `_template_preprocess.ipynb` 또는 `preprocess` 브랜치의 내 노트북 | `BRANCH = "preprocess"`, `NOTEBOOK = "notebooks/1_preprocess/<작업명>.ipynb"` |
| 새 모델 | `_template_model.ipynb` | `BRANCH = "model/knn"`, `NEW_BRANCH = True` |
| 기존 모델 수정 | `model/knn` 브랜치의 `knn.ipynb` | `BRANCH = "model/knn"`, `NEW_BRANCH = False` |
| 문서·공통 코드 | 로컬에서 하는 것을 권장 (3-2) | — |
| 코드 보기만 | `check_connection.ipynb` | `BRANCH = "main"` |

### 3-2. 내 PC에서

처음 한 번 [연결 튜토리얼 6절](SETUP_GUIDE.md#6-내-pc로컬-폴더-연결-선택)로 저장소를 받는다. 작업할 때마다:

```bash
git status                       # 고치고 안 올린 파일이 없는지 확인
git fetch --prune origin

# 문서·공통 / 전처리 / 기존 모델: 이미 있는 브랜치로
git switch common                # 또는 preprocess, model/knn
git pull --ff-only

# 새 모델만: main에서 새 브랜치
git switch main
git pull --ff-only
git switch -c model/svm
```

작업 후 올리기:

```bash
git status                       # 바뀐 파일 확인
git add docs/SETUP_GUIDE.md      # 파일을 직접 골라 담기
git diff --cached                # 담긴 내용 확인
git commit -m "docs: 토큰 승인 대기 해결 방법 추가"
git pull --rebase                # 같은 브랜치에 다른 사람이 먼저 올린 것을 합침
git push                         # 새 모델 브랜치를 처음 올릴 때는 git push -u origin model/svm
```

`git add .`는 데이터·모델 파일이 함께 들어갈 수 있어 쓰지 않는다.

### 3-3. 커밋 메시지

`<대상>: <무엇을 왜>` 형식으로 짧게 쓴다.

```text
knn: 거리 가중치(weights=distance) 비교 추가
crop: 상자 여백 10%로 변경 (작은 물건이 잘림)
docs: 연결 튜토리얼에 토큰 승인 대기 해결 방법 추가
```

실험 수치는 실제로 실행한 결과가 있을 때만 적는다.

## 4. PR로 main에 합치기

1. GitHub 저장소 → **Pull requests → New pull request** → base `main` ← compare `내 브랜치`(`common`, `preprocess`, `model/knn` 중 하나). `colab.push()` 출력의 PR 링크를 눌러도 된다.
2. 본문에는 [PR 틀](../.github/pull_request_template.md)이 자동으로 채워진다. 무엇을 왜 바꿨는지, 바꾼 파일, 실제로 확인한 것과 확인하지 못한 것을 적고 점검 항목에 체크한다.
3. 같은 브랜치에 이미 열린 PR이 있으면 새로 열지 않는다. 브랜치에 올리면 열린 PR에 자동으로 추가된다.
4. **다른 팀원 한 명 이상이 확인**한 뒤 합친다. AI가 작성했어도 사람의 확인을 대신하지 않는다.
5. 병합 버튼은 **Create a merge commit**을 쓴다. 브랜치를 계속 쓰기 때문에 Squash·Rebase로 합치면 다음 PR에서 같은 내용이 다시 충돌한다.
6. 병합 후 **Delete branch 버튼을 누르지 않는다.**

## 5. AI 에이전트에게 맡기기

Claude Code·Codex 등은 저장소의 [AGENTS.md](../AGENTS.md)를 읽고 같은 규칙을 따른다. 맡길 때는 **브랜치·범위·성공 기준**을 함께 말한다.

**문서·공통 작업**

> AGENTS.md와 docs/COLLABORATION.md를 읽고 `common` 브랜치로 바꿔 최신 내용을 받아 줘. SETUP_GUIDE.md에 ○○ 내용을 추가하고, 커밋·push한 뒤 main으로 가는 PR이 없으면 만들어 줘. 병합은 하지 마.

**새 모델 추가**

> AGENTS.md를 읽고, 먼저 `git fetch`로 `model/knn`이 있는지 확인해 줘. 없으면 main에서 `model/knn`을 새로 만들고 `src/models/knn.py`와 `notebooks/2_models/knn.ipynb`만 만들어. 다른 모델·전처리·공통 코드는 건드리지 마.

**기존 모델 수정**

> `model/knn` 브랜치로 바꾸고 최신 내용을 받아 줘. KNN에 거리 가중치 옵션을 추가하고, 이 브랜치의 KNN 파일만 고쳐. 커밋·push하고 `model/knn` → main PR을 만들어 줘. 데이터가 없으면 실행하지 못한 부분을 따로 알려 줘.

**PR 검토**

> PR #12를 읽고 데이터 누수·분할·지표가 docs/CONTRACTS.md와 맞는지, 브랜치 범위를 벗어난 파일이 있는지 검토해 줘.

주의:
- 에이전트가 `main`에서 작업하거나, 새 `docs/`·`fix/` 브랜치를 만들거나, 다른 모델 브랜치의 파일을 고치려 하면 멈추게 한다.
- 에이전트가 말한 실험 수치·참고문헌은 실제 실행 기록과 대조해 확인한다.
- 토큰·API 키를 에이전트 대화에 붙여 넣지 않는다.

## 6. 충돌과 실수 복구

| 상황 | 이렇게 한다 |
| --- | --- |
| Colab에서 `자동으로 합칠 수 없습니다` | 런타임을 끄지 않는다. 같은 브랜치를 쓰는 팀원과 누가 어떤 부분을 고쳤는지 확인하고, 한 사람이 로컬에서 합친다 |
| 로컬 `git pull`에서 충돌 | 충돌 파일을 열어 `<<<<<<<`·`=======`·`>>>>>>>` 사이의 양쪽 내용을 확인하고 합친 뒤 `git add` → `git commit`. 모르겠으면 `git merge --abort`(rebase 중이면 `git rebase --abort`)로 되돌리고 팀원에게 묻는다 |
| 노트북(.ipynb) 충돌 | 파일 전체를 한쪽으로 덮지 않는다. 양쪽 셀을 확인해 합치고, 재시작 후 위에서부터 실행해 본다 |
| 실수로 `main`에 커밋함 (아직 push 전) | 팀원에게 알리고, 그 커밋을 맞는 브랜치(`common` 등)로 옮긴다. 모르면 AI에게 "이 커밋을 common으로 옮겨 줘" |
| 다른 모델 브랜치에서 작업해 버림 | 올리기 전에 알리고, 맞는 브랜치로 옮긴다 |
| 규칙 밖 브랜치를 만들어 버림 | 내용을 맞는 브랜치로 옮긴 뒤 그 브랜치를 삭제한다 |
| 토큰·키를 커밋함 | **즉시 GitHub에서 토큰을 삭제(Revoke)** 하고 팀장에게 알린다. 커밋을 지워도 이미 노출된 것으로 본다 |

다른 사람의 작업을 `git reset --hard`, `git clean`, `git push --force`로 지우지 않는다.

## 7. 인수인계와 제출

작업을 넘길 때 브랜치·PR 링크, 바꾼 파일, 실제 실행한 노트북과 환경(Colab GPU 종류 등), 필요한 데이터 위치(Drive 경로), 결과와 남은 문제를 PR이나 팀 대화방에 남긴다. 비공개 데이터나 인증 정보는 붙이지 않는다.

제출 전에는 README의 실행 순서를 새 Colab 런타임에서 처음부터 확인한다. 저장소 공개 전환·초대는 팀장이 결정한다. 수업 일정은 [공식 안내](https://github.com/2026-2nd)를 확인한다.

## 8. 참고

- [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow)
- [PR 병합 방식](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges)
- [Fine-grained personal access token 만들기](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
- [Colab에서 GitHub 노트북 열기](https://colab.research.google.com/github/)
