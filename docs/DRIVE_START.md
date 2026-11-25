# 0. 시작하기 — 26-2 Team 1 (분리배출 안내 모델)

이 파일은 Google Drive `ML_TeamWork` 폴더의 첫 안내문입니다.
원본은 GitHub의 docs/DRIVE_START.md 이며, 노트북을 실행하면 자동으로 최신 내용으로 바뀝니다. (여기서 고치지 마세요.)


## 1. 이 폴더에 있는 것

    ML_TeamWork/
      0_시작하기.md     ← 지금 보는 파일
      code/            GitHub main 코드의 '보기용' 사본 (자동 갱신, 여기서 고치지 않기)
      data/raw/        원본 데이터 (AI Hub, 절대 수정 금지)
      data/processed/  전처리 결과
      models/          학습한 모델 파일
      outputs/         그 밖의 큰 출력물

코드의 기준은 GitHub 입니다: https://github.com/2026-2nd/26-2-1
코드는 Colab 노트북이나 내 PC의 git으로 고치고 GitHub에 올립니다. Drive에는 데이터·모델만 저장합니다.


## 2. 처음 한 번: 연결하기 (약 20분)

아래 튜토리얼을 위에서부터 순서대로 따라 하세요.

    연결 튜토리얼: https://github.com/2026-2nd/26-2-1/blob/main/docs/SETUP_GUIDE.md
    (Drive 안에서 보려면: code/docs/SETUP_GUIDE.md)

요약:
  ① GitHub 저장소 초대 수락 (메일 또는 https://github.com/2026-2nd/26-2-1)
  ② 이 ML_TeamWork 폴더를 '내 드라이브'에 바로가기로 추가
       drive.google.com → 공유 문서함 → ML_TeamWork 오른쪽 클릭 → 정리 → 바로가기 추가 → 내 드라이브
       (내 드라이브 바로 아래, 이름 그대로 둘 것)
  ③ GitHub 토큰 만들기 → Colab 🔑 Secrets에 GITHUB_TOKEN 으로 등록 (튜토리얼 3절)
  ④ 연결 확인 노트북을 Colab에서 열고 위에서부터 실행 → 4번 셀에서 ✅ 가 나오면 끝
       https://colab.research.google.com/github/2026-2nd/26-2-1/blob/main/notebooks/0_setup/check_connection.ipynb


## 3. 읽을 문서 (순서대로)

  1. 연결 튜토리얼   docs/SETUP_GUIDE.md      Colab·Drive·GitHub·내 PC 연결, 매일 작업하는 법, 자주 나는 오류
  2. 협업 규칙      docs/COLLABORATION.md    브랜치·PR 규칙, 충돌 났을 때, AI에게 맡기는 법
  3. 프로젝트 소개   README.md                주제·계획·폴더 구조
  4. 데이터 규칙     docs/DATA_SOURCES.md     AI Hub 데이터 이용 조건 (데이터를 Git·카톡·외부에 올리지 않기)
  5. AI 도구 쓸 때   AGENTS.md                Claude Code·Codex 등이 따르는 규칙
  (실험할 때)       docs/ML_WORKFLOW.md, docs/CONTRACTS.md

모두 GitHub(https://github.com/2026-2nd/26-2-1) 또는 이 폴더의 code/ 안에서 볼 수 있습니다.


## 4. 꼭 지킬 규칙

브랜치는 세 종류뿐입니다.

    common           문서·공통 코드·설정·오류 수정   → 모두 이 브랜치 하나에서
    preprocess       전처리                         → 모두 이 브랜치 하나에서
    model/<모델명>    모델 하나 (예: model/knn)       → 새 모델을 추가할 때만 새로 만들기
                                                      기존 모델 수정은 그 모델 브랜치에서

  - main 에는 직접 올리지 않고 PR 로만 합칩니다. 병합은 'Create a merge commit', 'Delete branch'는 누르지 않기.
  - 노트북: 첫 셀(BRANCH 지정) → 작업 → 마지막 셀(colab.push)로 GitHub에 올리기.
  - Colab 런타임이 끊기면 올리지 않은 코드는 사라집니다. 자주 올리세요.
  - 토큰·API 키는 노트북·카톡·문서에 붙여 넣지 않습니다.
  - 데이터·모델 파일은 Git에 올리지 않고 이 Drive 폴더에 둡니다.


## 5. 막히면

  - 튜토리얼 8절 '자주 나는 문제' 표에서 오류 메시지를 찾아보세요.
  - 그래도 안 되면 오류 메시지 전체(토큰 제외)를 팀 단톡방에 올려 주세요.
