# 협업 규칙 — 브랜치 · 커밋 · PR · AI 에이전트

Git으로 팀 프로젝트를 처음 하는 팀원을 위한 규칙이다. Colab·Drive·GitHub·내 PC 연결은 [연결 튜토리얼](SETUP_GUIDE.md)을 먼저 따라 한다.

대상 저장소는 `2026-2nd/26-2-1`(Public), 기본 브랜치는 `main`이다. 이 문서는 GitHub 설정(브랜치 보호·필수 리뷰 등)을 자동으로 바꾸지 않는다. 담당자는 아직 정하지 않았다.

## 1. 한눈에 보는 규칙

1. **`main`은 모두가 쓰는 완성본이다. 직접 고치지 않고 PR로만 합친다.**
2. **새 모델 = 새 브랜치.** KNN을 추가하면 반드시 `model/knn`을 새로 만든다.
3. **그 모델의 수정은 그 브랜치 안에서만.** KNN을 고칠 때는 언제나 `model/knn`에서 한다. 다른 모델 브랜치에서 KNN 파일을 고치지 않는다.
4. **브랜치를 만들기 전에 목록을 확인한다.** 같은 모델·작업의 브랜치가 있으면 새로 만들지 않고 그 브랜치를 쓴다.
5. **데이터·모델 파일·토큰은 Git에 올리지 않는다.** 데이터·모델은 Drive, 토큰은 Secrets.

## 2. 브랜치 규칙

### 2-1. 이름: `<종류>/<이름>`

`/` 뒤에는 영어 소문자·숫자·`-`·`_`만 쓴다. Colab 도우미(`colab.setup`, `colab.push`)는 이 형식이 아니면 멈춘다.

| 종류 | 언제 | 예 | 수명 |
| --- | --- | --- | --- |
| `model/` | 모델 하나 | `model/knn`, `model/svm`, `model/cnn` | **모델마다 하나, 계속 사용** |
| `preprocess/` | 전처리 작업 하나 | `preprocess/crop`, `preprocess/color-features` | 작업마다 하나, 계속 사용 |
| `setup/` | Colab·Drive·Git 연결, 환경, 공통 코드(`src/common/`) | `setup/colab-drive-git` | 병합 후 삭제 |
| `docs/` | 문서만 수정 | `docs/setup-guide` | 병합 후 삭제 |
| `fix/` | 이미 `main`에 있는 코드의 오류 수정 | `fix/paths-windows` | 병합 후 삭제 |

`junhyuk-test`, `new`, `final2`처럼 개인 이름이나 의미 없는 이름의 브랜치는 만들지 않는다.

### 2-2. 모델 브랜치가 다룰 수 있는 파일

`model/knn` 브랜치에서 고치는 파일은 다음뿐이다.

```text
notebooks/2_models/knn*.ipynb
src/models/knn.py           (필요하면 src/models/knn/ 폴더)
reports/experiments/*_knn/  (실험 기록)
```

다른 모델 파일, 전처리 코드, 공통 코드(`src/common/`), 문서를 고쳐야 하면 **그 일에 맞는 브랜치**(`preprocess/...`, `setup/...`, `docs/...`, `fix/...`)를 따로 만든다. 그래야 PR 하나가 한 가지 일만 담고, 모델끼리 충돌하지 않는다.

### 2-3. 모델 브랜치의 흐름

```text
main ──●──────────────●────────────────●──────▶
        \            ↑ PR #1 병합       ↑ PR #2 병합
         \ model/knn  │                 │
          ●──●──●─────┘──●──●───────────┘──●──▶  (KNN 작업은 계속 여기서)
          첫 버전         k 값 튜닝            ...
```

- 첫 버전이 되면 PR을 열어 `main`에 합친다. **병합 후에도 `model/knn` 브랜치는 지우지 않는다.** GitHub의 'Delete branch' 버튼은 누르지 않는다.
- 이후 KNN 수정도 같은 `model/knn`에서 하고, 다시 PR을 연다.
- 다른 브랜치에서 `main`이 바뀌어 내 브랜치가 뒤처지면 PR 화면의 **Update branch** 버튼으로 `main`의 변경을 받는다. 로컬에서는 `git merge origin/main`.

### 2-4. 브랜치 목록 확인

- Colab: [연결 확인 노트북](../notebooks/0_setup/check_connection.ipynb)의 `colab.branches()`
- GitHub: 저장소 → **Branches**
- 로컬: `git fetch --prune origin` 후 `git branch -r`

## 3. 작업 순서

### 3-1. Colab에서 (권장)

[연결 튜토리얼 5절](SETUP_GUIDE.md#5-colab에서-일하는-법-매번)대로 한다. 요약하면 이렇다.

| 상황 | 첫 셀 | 그다음 |
| --- | --- | --- |
| 새 모델 시작 | `BRANCH = "model/knn"`, `NEW_BRANCH = True` (템플릿에서) | 작업 → 마지막 셀 `colab.push(...)` |
| 이어서 작업 | GitHub 탭에서 `model/knn` 브랜치의 노트북을 열고 `NEW_BRANCH = False` | 작업 → 마지막 셀 |
| 코드 보기만 | `BRANCH = "main"` | 올리지 않음 |

### 3-2. 내 PC에서

처음 한 번 [연결 튜토리얼 6절](SETUP_GUIDE.md#6-내-pc로컬-폴더-연결-선택)로 저장소를 받는다. 작업할 때마다:

```bash
git status                       # 고치고 안 올린 파일이 없는지 확인
git fetch --prune origin

# (A) 새 모델 시작: main에서 새 브랜치
git switch main
git pull --ff-only
git switch -c model/knn

# (B) 이미 있는 브랜치로 이어서
git switch model/knn
git pull --ff-only
```

작업 후 올리기:

```bash
git status                       # 바뀐 파일 확인
git add src/models/knn.py notebooks/2_models/knn.ipynb    # 파일을 직접 골라 담기
git diff --cached                # 담긴 내용 확인
git commit -m "knn: k=1~15 비교 추가"
git push -u origin model/knn     # 처음 한 번 -u, 이후에는 git push
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

1. 브랜치를 올리면 `colab.push()` 출력이나 GitHub 상단 배너에 PR 링크가 나온다. 또는 저장소 → **Pull requests → New pull request** → base `main` ← compare `model/knn`.
2. 본문에는 [PR 틀](../.github/pull_request_template.md)이 자동으로 채워진다. 무엇을 왜 바꿨는지, 바꾼 파일, 실제로 확인한 것과 확인하지 못한 것을 적고 점검 항목에 체크한다.

3. **다른 팀원 한 명 이상이 확인**한 뒤 합친다. AI가 작성했어도 사람의 확인을 대신하지 않는다.
4. 충돌이 있으면 6절을 따른다. 병합 방식·병합할 사람은 팀에서 정한다.
5. `model/`·`preprocess/` 브랜치는 병합 후에도 지우지 않는다. `setup/`·`docs/`·`fix/`는 병합 후 삭제한다.

## 5. AI 에이전트에게 맡기기

Claude Code·Codex 등은 저장소의 [AGENTS.md](../AGENTS.md)를 읽고 같은 규칙을 따른다. 맡길 때는 **브랜치·범위·성공 기준**을 함께 말한다.

**새 모델 브랜치 만들기**

> AGENTS.md와 docs/COLLABORATION.md를 읽고, main에서 `model/knn` 브랜치를 새로 만들어 줘. 먼저 `git fetch`로 같은 이름의 브랜치가 없는지 확인해. `src/models/knn.py`와 `notebooks/2_models/knn.ipynb`만 만들고, 다른 모델·공통 코드는 건드리지 마.

**이어서 고치기**

> `model/knn` 브랜치로 바꾸고 최신 내용을 받아 줘. KNN에 거리 가중치 옵션을 추가하고, 이 브랜치의 파일만 고쳐. 데이터가 없으면 실행하지 못한 부분을 따로 알려 줘.

**PR 만들기**

> 지금 브랜치의 변경을 확인하고 커밋한 뒤 push하고, main으로 가는 PR을 만들어 줘. PR 본문은 .github/pull_request_template.md 틀을 채우고, 실행하지 않은 결과는 적지 마. 병합은 하지 마.

**PR 검토 받기**

> PR #12를 읽고 데이터 누수·분할·지표가 docs/CONTRACTS.md와 맞는지, 이 브랜치 범위를 벗어난 파일이 있는지 검토해 줘.

주의:
- 에이전트가 `main`에서 작업하거나 다른 모델 브랜치의 파일을 고치려 하면 멈추게 한다.
- 에이전트가 말한 실험 수치·참고문헌은 실제 실행 기록과 대조해 확인한다.
- 토큰·API 키를 에이전트 대화에 붙여 넣지 않는다.

## 6. 충돌과 실수 복구

| 상황 | 이렇게 한다 |
| --- | --- |
| Colab에서 `자동으로 합칠 수 없습니다` | 런타임을 끄지 않는다. 같은 브랜치를 쓰는 팀원과 누가 어떤 부분을 고쳤는지 확인하고, 한 사람이 로컬에서 합친다 |
| 로컬 `git pull`에서 충돌 | 충돌 파일을 열어 `<<<<<<<`·`=======`·`>>>>>>>` 사이의 양쪽 내용을 확인하고 합친 뒤 `git add` → `git commit`. 모르겠으면 `git merge --abort`로 되돌리고 팀원에게 묻는다 |
| 노트북(.ipynb) 충돌 | 파일 전체를 한쪽으로 덮지 않는다. 양쪽 셀을 확인해 합치고, 재시작 후 위에서부터 실행해 본다 |
| 실수로 `main`에 커밋함 (아직 push 전) | `git switch -c fix/<이름>`으로 그 커밋을 새 브랜치로 옮긴 뒤 팀원에게 알린다 |
| 다른 모델 브랜치에서 작업해 버림 | 올리기 전에 알리고, 맞는 브랜치로 옮긴다 (필요하면 AI에게 "커밋을 `model/knn`으로 옮겨 줘") |
| 토큰·키를 커밋함 | **즉시 GitHub에서 토큰을 삭제(Revoke)** 하고 팀장에게 알린다. 커밋을 지워도 이미 노출된 것으로 본다 |

다른 사람의 작업을 `git reset --hard`, `git clean`, `git push --force`로 지우지 않는다.

## 7. 인수인계와 제출

작업을 넘길 때 브랜치·PR 링크, 바꾼 파일, 실제 실행한 노트북과 환경(Colab GPU 종류 등), 필요한 데이터 위치(Drive 경로), 결과와 남은 문제를 PR이나 팀 대화방에 남긴다. 비공개 데이터나 인증 정보는 붙이지 않는다.

제출 전에는 README의 실행 순서를 새 Colab 런타임에서 처음부터 확인한다. 저장소 공개 전환·초대는 팀장이 결정한다. 수업 일정은 [공식 안내](https://github.com/2026-2nd)를 확인한다.

## 8. 참고

- [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow)
- [Fine-grained personal access token 만들기](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
- [Colab에서 GitHub 노트북 열기](https://colab.research.google.com/github/)
