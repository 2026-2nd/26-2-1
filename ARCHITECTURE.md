# 프로젝트 구조 안내 — 26-2-1

이 문서는 데이터에서 실험 결과까지의 흐름과 파일을 둘 위치를 설명한다.

## 1. 현재 상태

- 등록된 항목: 제안서, README·AI 지침·협업·실험·공통 기준·데이터 출처·연결 튜토리얼 문서, Colab·Drive·GitHub 연결 코드(`src/common/`), AI Hub 다운로드 도구(`src/preprocess/aihub_download.py`), 연결 확인·전처리·모델 템플릿 노트북, 단위 테스트.
- 제안된 주제: 단일 물건 사진의 품목을 다중 분류한 뒤 규칙표와 연결하는 분리배출 안내 PoC. 모델·실험 계획은 [README](README.md)를 따른다.
- 적용 범위: 오염·라벨 등 상태 판별은 포함하지 않는다. 안내에는 ‘내용물을 비우고 헹군 상태’를 전제로 한다는 문구를 함께 표시할 계획이다.
- 아직 없음: 전처리·모델 코드, 실험 결과, CI. 2026-11-23에 만든 이전 파이프라인 코드는 팀이 처음부터 다시 만들기로 해 제거했다 (`feat/verdict-pipeline` 브랜치에 기록이 남아 있다).
- 미정: 세부 클래스·팀원 역할.
- 저장소는 `2026-2nd/26-2-1` 조직 소유의 Public 저장소이다. 공개 범위와 팀원 초대는 사용자의 요청에 따라 별도로 관리한다.

## 2. 문서 지도

| 문서 | 내용 |
| --- | --- |
| [README.md](README.md) | 프로젝트 소개, 실행 방법과 결과 요약 |
| [AGENTS.md](AGENTS.md) | AI 도구 공통 작업 규칙 |
| [CLAUDE.md](CLAUDE.md) | Claude Code에서 공통 규칙을 읽는 진입점 |
| [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | Colab·Drive·GitHub·내 PC 연결 튜토리얼 |
| [docs/COLLABORATION.md](docs/COLLABORATION.md) | 브랜치·커밋·PR·충돌 해결·AI 에이전트 |
| [docs/ML_WORKFLOW.md](docs/ML_WORKFLOW.md) | 데이터 확보부터 최종 평가까지의 실험 절차 |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | 데이터·분할·지표·결과 기록의 공통 기준 |
| [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) | 데이터 출처·이용 조건·다운로드 기록 |

## 3. 폴더

단계(전처리·모델)별로 노트북과 코드를 같은 이름으로 나눈다. 파일·폴더 이름은 import와 경로 문제를 피하려고 영어로 쓰고, 노트북 안의 제목·설명은 한국어로 쓴다.

```text
26-2-1/
  docs/
  notebooks/
    0_setup/check_connection.ipynb      # 연결 확인 (처음 한 번)
    1_preprocess/                       # 전처리: <작업명>.ipynb  ← preprocess/<작업명> 브랜치
      _template_preprocess.ipynb
    2_models/                           # 모델: <모델명>.ipynb    ← model/<모델명> 브랜치
      _template_model.ipynb
  src/
    common/        # 모든 단계 공통 — paths.py(위치), colab.py(Drive·Git 연결)
    preprocess/    # 전처리 코드 — <작업명>.py, aihub_download.py
    models/        # 모델 코드 — <모델명>.py (모델 하나 = 파일 하나)
  configs/         # 데이터·분할·모델 설정 (aihub_71385.json)
  reports/experiments/   # 실행별 기록 (필요할 때 생성, Git에 올림)
  tests/
  .github/pull_request_template.md

Google Drive  내 드라이브/ML_TeamWork/   (Git 제외)
  data/raw/        # 원본, 직접 수정하지 않음
  data/processed/  # 코드로 다시 만드는 전처리 결과
  models/          # 학습한 모델 파일
  outputs/         # 그 밖의 큰 출력물
```

노트북의 첫 셀은 `colab.setup(BRANCH, ...)`로 GitHub 코드를 `/content/26-2-1`에 받고 Drive를 연결하며, 마지막 셀은 `colab.push(MESSAGE)`로 노트북과 코드 변경을 그 브랜치에 올린다. Drive 안에 저장소를 복제해 두지 않는다. 웹앱·API·별도 실험 관리 서비스는 현재 필수 구성으로 정하지 않는다.

## 4. 데이터와 실행 흐름

```text
출처·버전 확인 → 원본 보존 → 스키마·품질 확인 → 공통 분할 확정
→ train에서 전처리·모델 학습 → validation 또는 CV로 비교·선택
→ 설정 확정 후 test 평가 → 결과·그림·해석 → 보고서
```

- 원본 로딩과 컬럼 정리는 입력 단계가 맡는다. 중복·그룹·시간 정보는 분할 전에 확인한다.
- 스케일링·결측 대체·피처 선택 등 학습이 필요한 단계는 학습 데이터에만 맞춘다. scikit-learn 사용 시 `Pipeline`과 `ColumnTransformer`를 활용한다.
- 학습 코드와 노트북은 같은 전처리·평가 함수를 재사용한다. 같은 이름의 지표를 서로 다르게 계산하지 않는다.
- 시각화는 기록된 실험 결과를 사용한다. 그래프 제목·축·표본 수·평가셋을 명확히 쓴다.

추론 단계는 ‘단일 물건 사진 → 품목 분류 → 규칙표 연결 → 전제 조건을 포함한 배출 안내’를 계획한다. 확신이 낮으면 판단을 보류하고 재촬영을 요청한다. 품목 분류 결과와 물건의 오염·라벨 상태를 혼동하지 않는다.

## 5. 구성 요소 사이의 접점

| 접점 | 함께 관리할 내용 |
| --- | --- |
| 데이터 → 전처리 | 파일 버전, 행의 단위, ID, 컬럼·자료형, 타깃, 누락값 정책 |
| 전처리 → 학습 | 피처 목록·순서, 인코딩, fit 범위, 학습/검증/테스트 ID |
| 학습 → 평가 | 클래스·확률 열의 순서, 예측값 단위, 임계값, 지표 정의 |
| 품목 분류 → 배출 안내 | 예측 품목, 판단 보류 여부, 규칙표 버전·공식 출처, ‘내용물을 비우고 헹군 상태’ 전제 문구 |
| 평가 → 보고서 | 실행 ID, 코드 버전, 데이터·분할 버전, 실제 점수와 한계 |

구체적인 값은 [공통 기준 문서](docs/CONTRACTS.md)에 확정한다. 위 표는 인터페이스 계획이며 현재 함수가 있다는 뜻은 아니다. 함수를 만들면 같은 PR에서 이 표와 CONTRACTS를 갱신한다.

## 6. 실행 환경과 경로

주 실행 환경은 Colab이며 패키지는 Colab 기본 설치본을 쓴다. 로컬에서 확인한 버전은 `requirements.txt`에 있다. 실험에 쓴 실제 버전은 실험 기록에 남긴다.

경로는 `src/common/paths.py`로만 얻는다 (`paths.data_dir()`, `paths.processed_dir()`, `paths.models_dir()`, `paths.run_dir(name)`). 위치는 환경변수 `ML_TEAMWORK_DIR` → Colab의 Drive(`/content/drive/MyDrive/ML_TeamWork`) → 저장소 폴더 순서로 정해진다. 개인 PC·Drive의 절대 경로를 코드에 적지 않는다.

데이터·모델·가상환경·비밀키는 `.gitignore`로 제외되며, `colab.push()`는 5MB보다 큰 파일을 거부한다. 작은 공개 샘플을 Git에 넣을 경우에는 사용 조건과 포함 이유를 기록한다.

## 7. 구조를 갱신할 때

폴더·실행 진입점·공유 함수·환경이 달라지면 이 문서와 README를 함께 갱신한다. 새 코드가 생기면 ‘현재 상태’도 수정한다. 샘플 명령을 실제 실행이 검증된 명령처럼 기재하지 않는다.
