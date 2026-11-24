# 연결 튜토리얼 — Colab · Google Drive · GitHub · 내 PC

처음 프로젝트에 참여하는 팀원이 **위에서부터 순서대로** 따라 하는 문서다. 브랜치·PR 규칙은 [협업 규칙](COLLABORATION.md)에 있다.

## 0. 전체 그림 — 무엇이 어디에 있나

```text
              ┌──────────── GitHub (2026-2nd/26-2-1) ────────────┐
              │  코드·노트북·문서·작은 실험 기록. 모든 복사본의 기준  │
              └───────▲─────────────────────────────▲────────────┘
     colab.push() / 받기│                    git push / pull│
┌──────────────────────┴──────┐        ┌─────────────────┴──────────┐
│ Colab 런타임                  │        │ 내 PC (선택)                 │
│ /content/26-2-1 ← 코드 복사본 │        │ C:\...\26-2-1 ← 코드 복사본   │
│ (런타임이 끝나면 사라짐)       │        │ VS Code 등으로 편집           │
└──────────────┬──────────────┘        └───────────────┬────────────┘
     Drive 연결 │ 데이터 읽기·쓰기          Drive 데스크톱 │ (선택)
              ┌▼─────────────────────────────────────▼┐
              │ Google Drive  내 드라이브/ML_TeamWork/   │
              │   data/raw/        원본 데이터 (수정 금지) │
              │   data/processed/  전처리 결과           │
              │   models/          학습한 모델 파일       │
              │   outputs/         그 밖의 큰 출력물      │
              └────────────────────────────────────────┘
```

| 무엇 | 어디에 | Git에 올리나 |
| --- | --- | --- |
| 코드(`src/`), 노트북(`notebooks/`), 문서, 설정(`configs/`) | GitHub | ✅ |
| 실험 기록(`reports/experiments/`의 설정·점수·그래프) | GitHub | ✅ (데이터 사진이 들어간 그림은 ❌, [데이터 출처](DATA_SOURCES.md) 1번 규칙) |
| 원본·전처리 데이터, 모델 파일, 큰 출력물 | Drive `ML_TeamWork/` | ❌ |
| API 키·GitHub 토큰 | Colab 🔑 Secrets, 내 PC 환경변수 | ❌ 절대 안 됨 |

핵심 원칙:
- **코드의 기준은 GitHub 하나다.** Colab과 내 PC에 있는 것은 복사본이고, 고친 내용은 GitHub에 올려야 다른 사람에게 보인다.
- **Drive에는 데이터만 둔다.** Drive 안에 Git 저장소를 복사해 두지 않는다 (동기화 충돌·`desktop.ini` 문제).
- **`main`에는 직접 올리지 않는다.** 작업은 브랜치에서 하고 PR로 합친다.

## 1. 준비물 (한 번)

1. **GitHub 계정** — 팀장에게 아이디를 알려 주고, 메일 또는 <https://github.com/2026-2nd/26-2-1>에 뜨는 **초대를 수락**한다. 수락해야 쓰기 권한이 생긴다.
2. **Google 계정** — Colab과 Drive에 쓴다.
3. **AI Hub 데이터 이용 승인** — 데이터는 각자 AI Hub에서 신청해 승인받는다. 승인받은 사람에게만 Drive 데이터 폴더를 공유한다 ([데이터 출처](DATA_SOURCES.md)).

## 2. Google Drive 연결 (한 번)

팀 데이터는 `ML_TeamWork` 폴더 하나에 모은다.

1. 공유 초대를 받으면 <https://drive.google.com> → 왼쪽 **공유 문서함** → `ML_TeamWork` 폴더를 찾는다.
2. 폴더를 오른쪽 클릭 → **정리** → **바로가기 추가** → **내 드라이브** 선택.
3. **내 드라이브** 바로 아래에 `ML_TeamWork`가 보이면 끝. (Colab에서 `/content/drive/MyDrive/ML_TeamWork`로 열린다.)

> 바로가기를 다른 폴더 안에 넣거나 이름을 바꾸면 노트북이 찾지 못한다. 꼭 **내 드라이브 바로 아래, 이름 그대로** 둔다.

## 3. GitHub 토큰 만들기 + Colab에 등록 (한 번)

Colab에서 GitHub로 올리려면(`colab.push()`) 본인 토큰이 필요하다. 토큰은 비밀번호와 같으므로 **노트북·코드·카톡·문서에 붙여 넣지 않는다.**

### 3-1. 토큰 만들기 (권장: Fine-grained token)

1. GitHub 오른쪽 위 프로필 → **Settings** → 왼쪽 맨 아래 **Developer settings** → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**.
2. 다음처럼 채운다.

   | 항목 | 값 |
   | --- | --- |
   | Token name | `colab-26-2-1` 등 알아보기 쉬운 이름 |
   | Expiration | 학기 종료일 정도 (무기한 금지) |
   | Resource owner | **`2026-2nd`** (본인 계정이 아님) |
   | Repository access | **Only select repositories** → `2026-2nd/26-2-1` |
   | Permissions → Repository permissions | **Contents: Read and write** (Metadata: Read는 자동) |

3. **Generate token** → 나온 값(`github_pat_...`)을 복사한다. 이 화면을 벗어나면 다시 볼 수 없다.

Resource owner에 `2026-2nd`가 안 보이거나 토큰이 '승인 대기(pending)'이면 조직 설정 때문이다. 팀장에게 알려 조직 관리자 승인을 요청한다. 그동안은 Classic token(`repo` 권한, 짧은 만료일)으로 대신할 수 있지만 권한 범위가 넓으므로 끝나면 삭제한다.

### 3-2. Colab Secrets에 등록

1. <https://colab.research.google.com>에서 아무 노트북이나 연다.
2. 왼쪽 세로 막대의 **🔑(Secrets)** 아이콘 → **새 보안 비밀 추가**.
3. 이름 `GITHUB_TOKEN`, 값에 토큰을 붙여 넣는다.
4. **노트북 액세스** 스위치를 켠다. (Secrets는 계정에 저장되며, 새 노트북을 열 때마다 이 스위치를 켜라는 창이 뜨면 허용한다.)

> Secrets에 없으면 `colab.push()`가 입력칸을 띄운다. 거기에 붙여 넣어도 노트북에는 저장되지 않는다.

## 4. Colab 첫 실행 — 연결 확인 (한 번)

1. GitHub에서 [`notebooks/0_setup/check_connection.ipynb`](../notebooks/0_setup/check_connection.ipynb)를 열고 **Open in Colab** 배지를 누른다.
   (또는 Colab → **파일 → 노트북 열기 → GitHub** 탭 → `2026-2nd/26-2-1` 입력 → 브랜치 `main` → 파일 선택.)
2. 셀을 위에서부터 하나씩 실행한다 (▶ 또는 `Shift+Enter`). Drive 권한 창은 **허용**.
3. 결과 확인:
   - 1번 셀: `브랜치: main`, 데이터·모델 경로가 `/content/drive/MyDrive/ML_TeamWork/...`
   - 2번 셀: Drive의 데이터 폴더 목록
   - 3번 셀: GitHub 브랜치 목록
   - 4번 셀: `✅ GitHub 연결 확인: <내 아이디> 계정으로 ... 올릴 수 있습니다.`

4번까지 통과하면 Colab·Drive·GitHub 연결이 끝난 것이다. 막히면 [8. 자주 나는 문제](#8-자주-나는-문제)를 본다.

## 5. Colab에서 일하는 법 (매번)

모든 노트북은 같은 모양이다.

| 셀 | 하는 일 | 내가 고칠 것 |
| --- | --- | --- |
| **첫 셀 (준비)** | GitHub에서 코드를 받고, Drive를 연결하고, 브랜치를 준비 | `BRANCH`, `NEW_BRANCH` |
| 가운데 셀 | 그 단계의 작업 (전처리 또는 모델) | 코드 작성 |
| **마지막 셀 (올리기)** | 노트북·고친 코드·실험 기록을 브랜치에 커밋하고 GitHub에 올림 | `MESSAGE` |

### 5-1. 새 모델 시작하기 (예: KNN)

1. [연결 확인 노트북](../notebooks/0_setup/check_connection.ipynb) 3번 셀(또는 GitHub의 **Branches** 페이지)에서 `model/knn`이 이미 있는지 본다. **있으면 새로 만들지 말고 5-2로.**
2. [`notebooks/2_models/_template_model.ipynb`](../notebooks/2_models/_template_model.ipynb)를 Colab에서 연다.
3. 첫 셀을 고치고 실행한다.
   ```python
   BRANCH = "model/knn"
   NEW_BRANCH = True
   ```
4. 작업한다. 여러 번 쓸 코드는 Colab 왼쪽 📁 파일 창에서 `/content/26-2-1/src/models/knn.py`를 만들어 옮긴다.
5. 마지막 셀에 `MESSAGE = "knn: 첫 버전"`을 적고 실행한다. 이 노트북이 `notebooks/2_models/knn.ipynb`로, 코드와 함께 `model/knn` 브랜치에 올라간다.

전처리도 같다. 템플릿은 [`notebooks/1_preprocess/_template_preprocess.ipynb`](../notebooks/1_preprocess/_template_preprocess.ipynb), 브랜치는 `preprocess/<작업명>`.

### 5-2. 이어서 작업하기

1. Colab → **파일 → 노트북 열기 → GitHub** 탭 → `2026-2nd/26-2-1` → 브랜치 **`model/knn`** → `notebooks/2_models/knn.ipynb`.
2. 첫 셀이 `BRANCH = "model/knn"`, `NEW_BRANCH = False`인지 확인하고 실행한다. 같은 브랜치에 팀원이 올린 최신 내용을 받아 온다.
3. 작업 → 마지막 셀로 올린다. 같은 브랜치에 다른 사람이 먼저 올린 커밋이 있으면 자동으로 합친 뒤 올린다.

### 5-3. main에 합치기 (PR)

`colab.push()`가 끝나면 PR 링크를 출력한다. 링크를 열어 내용을 적고 **Create pull request**. 자세한 절차는 [협업 규칙 4절](COLLABORATION.md#4-pr로-main에-합치기).

### 5-4. Colab에서 꼭 알아 둘 것

- **런타임이 끊기면 `/content`가 지워진다.** 올리지 않은 코드는 사라지므로 작업 단위마다 마지막 셀로 올린다. Drive에 저장한 데이터·모델은 남는다.
- 데이터·모델 파일은 `paths.processed_dir()`, `paths.models_dir()`처럼 **`paths`가 알려 주는 Drive 위치**에 저장한다. 경로를 직접 적지 않는다.
- 5MB보다 큰 파일은 `colab.push()`가 올리지 않는다. Drive에 둔다.
- GPU가 필요하면 **런타임 → 런타임 유형 변경 → GPU**.
- **파일 → Drive에 사본 저장**은 쓰지 않는다. 노트북은 `colab.push()`로 GitHub에 저장한다.

## 6. 내 PC(로컬 폴더) 연결 (선택)

VS Code로 코드를 편하게 고치거나, AI Hub 데이터를 받아야 할 때(AI Hub는 해외 IP인 Colab에서 다운로드를 막는다) 쓴다.

### 6-1. 설치

| 프로그램 | 용도 |
| --- | --- |
| [Git](https://git-scm.com/downloads) | 코드 받기·올리기. 설치 중 옵션은 기본값 |
| [VS Code](https://code.visualstudio.com/) (또는 GitHub Desktop) | 편집. 왼쪽 '소스 제어'로 브랜치·커밋 가능 |
| Python 3.12 | 로컬 실행·테스트 (Colab만 쓰면 생략 가능) |
| [Google Drive 데스크톱](https://www.google.com/drive/download/) | 데이터를 PC에서 읽거나 AI Hub에서 받아 Drive에 저장할 때 |

### 6-2. 저장소 받기

**Drive 폴더 밖**의 일반 폴더(예: `C:\Users\<나>\Projects`)에서 실행한다.

```bash
git clone https://github.com/2026-2nd/26-2-1.git
cd 26-2-1
git config user.name "<GitHub 아이디>"
git config user.email "<GitHub에 등록한 이메일 또는 noreply 주소>"
```

처음 `git push`할 때 GitHub 로그인 창이 뜨면 브라우저로 로그인한다 (Windows용 Git의 자격 증명 관리자).

### 6-3. Python 환경 (선택)

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS·Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### 6-4. Drive 데이터를 PC에서 쓰기

Drive 데스크톱을 설치하면 `G:\내 드라이브` 같은 드라이브가 생긴다 (글자는 PC마다 다르다). 터미널에서 데이터 위치를 알려 준다.

```powershell
$env:ML_TEAMWORK_DIR = "G:\내 드라이브\ML_TeamWork"      # PowerShell, 이 창에서만 유효
```

이제 `paths.data_dir()` 등이 Drive를 가리킨다. 경로를 코드에 직접 적지 않는다. AI Hub 다운로드는 [데이터 출처 문서](DATA_SOURCES.md)의 명령을 이 상태에서 실행한다.

### 6-5. 로컬에서 브랜치 작업

명령과 순서는 [협업 규칙 3절](COLLABORATION.md#3-작업-순서)을 따른다. Colab과 로컬을 번갈아 써도 된다. **바꾸기 전에 받고(pull), 끝나면 올린다(push)** 만 지키면 GitHub를 통해 맞춰진다.

## 7. 하면 안 되는 것

- 토큰·API 키를 노트북 셀·코드·문서·카톡에 붙여 넣기
- `main`에 직접 커밋·푸시, `git push --force`
- 데이터·모델 파일을 Git에 올리기, 원본(`data/raw/`)을 수정하기
- Drive 안에 저장소를 clone하기, 다른 모델의 브랜치·파일을 고치기
- AI Hub 해외 다운로드 제한을 VPN 등으로 우회하기

## 8. 자주 나는 문제

| 메시지·증상 | 원인과 해결 |
| --- | --- |
| `Drive에 ... ML_TeamWork 폴더가 없습니다` | 2절의 바로가기를 **내 드라이브 바로 아래**에 추가했는지 확인 |
| `GitHub에 'model/...' 브랜치가 없습니다` | 이름 오타 확인. 처음 만드는 거면 `NEW_BRANCH = True` |
| `'model/...'는 이미 GitHub에 있습니다` | 이미 있는 브랜치. `NEW_BRANCH = False`로 이어서 작업 |
| `브랜치 이름은 '<종류>/<이름>' 형식이어야 합니다` | 예: `model/knn`. `/` 뒤는 영어 소문자·숫자·`-`만 |
| `main에는 직접 올리지 않습니다` | 첫 셀의 `BRANCH`를 작업 브랜치로 바꾸고 첫 셀부터 다시 |
| `GitHub 토큰이 잘못됐거나 만료됐습니다` | 3절로 새 토큰을 만들어 Secrets 값을 바꿈 |
| `이 토큰으로 ... 저장소를 볼 수 없습니다` / `쓰기 권한이 없습니다` | 초대 수락 여부, 토큰의 Resource owner(`2026-2nd`)·저장소 선택·Contents 권한 확인. 승인 대기면 팀장에게 |
| `아직 올리지 않은 파일이 있습니다` | 브랜치를 바꾸기 전에 마지막 셀로 먼저 올림 |
| `자동으로 합칠 수 없습니다` (충돌) | **런타임을 끄지 말고** 팀원과 상의. [협업 규칙 6절](COLLABORATION.md#6-충돌과-실수-복구) |
| `5MB보다 큰 파일은 ...` | 그 파일을 Drive(`paths.models_dir()` 등)로 옮기고 다시 실행 |
| `노트북 내용을 읽지 못했습니다` | Colab의 **파일 → GitHub에 사본 저장**으로 같은 브랜치·경로에 직접 저장 |
| `ModuleNotFoundError: No module named 'src'` | 첫 셀을 실행하지 않았거나 런타임이 재시작됨. 첫 셀부터 다시 |
| AI Hub `HTTP 502 ... 해외에서의 데이터 다운로드를 제한` | Colab에서는 받을 수 없음. 6-4처럼 국내 PC에서 받기 |
