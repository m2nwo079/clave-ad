# CLAUDE.md — CLAVE-AD

이 파일은 저장소에 포함되는 공용 작업 규칙이다. 개인 설정(노션 링크, 로컬 경로, 인증 정보 위치, 외부 실행 ID, 개인 작업 방식)은 `CLAUDE.local.md`에 있으며, 그 파일은 커밋하지 않는다. 응답 방식은 사용자 전역 설정(`~/.claude/CLAUDE.md`)을 따른다.

## 1. 프로젝트 개요

**CLAVE-AD**: Claim-Level Audit and Verification of Explanations for Anomaly Detection

VLM이 내놓는 산업 불량 설명을 존재·위치·유형의 주장으로 분해하고, VLM과 독립된 외부 도구(CLIP 기반 이상 점수, 이상 맵, 유형별 예시 라이브러리)로 각 주장을 검증한다. 검증에 실패한 주장은 의심 영역을 확대해 한 번 다시 묻는다. 목표는 정답 마스크가 없는 운영 환경에서도 믿을 수 있는 설명만 가려내는 것이다.

- 노션 설계서와 그 하위의 연구 과정 기록이 단일 원천이다(링크는 `CLAUDE.local.md`). 설계서에는 확정된 설계를, 연구 과정 기록에는 과정을 남긴다.
- 설계서가 이 파일과 충돌하면 설계서를 따르고, 충돌 사실을 사용자에게 알린다.

## 2. 반드시 지킬 설계 원칙

1. **정답은 평가에만 쓴다.** 정답 마스크와 테스트 라벨은 `src/clave_ad/eval/`에서만 읽는다. 설명 생성, 검증, 재질의 코드가 정답에 접근하면 설계 위반이다. 운영 경로에 허용되는 입력은 정상 이미지와 유형별 예시 라이브러리뿐이다. 예외로 `src/clave_ad/library/`만 예시 라이브러리용으로 분리한 이미지의 유형 라벨과 단일 유형 여부를 읽을 수 있고, 라이브러리 크롭은 정답 마스크가 아니라 이상 맵으로 만든다.
2. **검증자는 VLM과 독립이다.** 검증 로직에 VLM 호출을 넣지 않는다(B3 자기 검증 비교 대상은 예외이며 `baselines/`에 둔다).
3. **추론은 한 번, 분석은 여러 번.** VLM과 탐지기 출력은 이미지 단위로 즉시 `cache/`에 저장하고, 이미 있는 항목은 다시 계산하지 않는다. 중간에 끊겨도 이어서 실행할 수 있어야 한다.
4. **무료 자원 안에서 수행한다.** 주 실험은 Kaggle T4 추론만으로 끝나야 한다.
5. **규칙은 결과를 보기 전에 정한다.** 설계서 13절의 고정 규칙(임계값, 형식 위반 처리, 정답 칸 변환)을 평가 지표를 본 뒤 바꾸려면, 먼저 사용자에게 알리고 연구 과정 기록을 남긴다.
6. **수치는 문서에 직접 쓰지 않는다.** 결과 수치는 `results/*.json`에만 저장하고, 보고서는 템플릿으로 자동 생성한다. 문서, 주석, 노션, 이 파일에는 파일명과 git 커밋만 적는다.
7. **실패한 결과도 남긴다.** 제안 방식이 대안보다 못한 결과도 지우지 않는다.
8. **주장 범위를 넘지 않는다.** 설계서가 정한 주장 범위를 넘지 않고, 신규성은 "검색 범위 내 미발견"으로만 표현한다.

## 3. 저장소 구조

```
configs/<stage>/          # yaml configs per stage
src/clave_ad/
  data/                   # dataset loaders, grid and mask utilities
  detectors/              # CLIP-based anomaly detector wrappers
  vlm/                    # prompts, structured output, parsing
  claims/                 # claim decomposition
  verifiers/              # existence, location, type verifiers
  requery/                # zoom-in re-query
  library/                # example library split (reads labels of library images only)
  baselines/              # B0 to B4 comparison methods
  eval/                   # ground-truth metrics (only place that reads ground truth)
scripts/<stage>/          # entry points per stage
scripts/kaggle/push.sh    # shared Kaggle upload, takes the stage name
notebooks/kaggle/<stage>.py  # Kaggle kernel per stage
data/                     # raw datasets (not tracked)
cache/                    # model outputs, anomaly maps, crops (not tracked)
results/<stage>/          # metric json files per stage (tracked)
docs/                     # local notes and fallback research log
.claude/settings.json     # secret file read guard (not tracked)
.claude/skills/           # reusable procedures as skills (not tracked)
.githooks/pre-commit      # personal data and secret commit guard (tracked)
```

실행 입구(`configs/`, `scripts/`, `notebooks/kaggle/`, `results/`)는 단계별로, 여러 단계가 함께 쓰는 코드는 역할별로(`src/clave_ad/`, `scripts/kaggle/`) 나눈다. `<stage>`는 단계 브랜치 이름의 하이픈을 밑줄로 바꾼 것이다(`stage/memory-test` → `memory_test`).

## 4. 현재 단계

`CLAUDE.local.md`의 "현재 단계" 절에 둔다. 작업을 시작할 때 읽고, 끝날 때 갱신한다.

## 5. 작업 절차

### 5.1 작업 시작 전

- 노션 설계서를 읽고 현재 버전과 단계를 확인한다.
- 사용자가 코드나 설계서를 직접 고쳤을 수 있으므로, 이전 작업 이후 달라진 부분을 먼저 파악한다.
- 설계를 바꾸는 결정은 코드보다 먼저 설계서 반영을 제안한다. 설계서의 "확인 필요" 항목은 검증 전까지 사실로 취급하지 않는다.
- 단계 계획은 `~/.claude/plans/`의 계획 파일로 관리한다.

### 5.2 역할 분담

- 기본: 사용자가 코드를 작성하고, Claude는 과제 안내, 코드 리뷰, 막힐 때 예시 코드를 제공한다. 사용자가 요청할 때만 Claude가 코드를 작성한다.
- 작업 범위가 불분명하면 실행 전에 역할을 먼저 확인한다. 묻지 않고 실행까지 진행하지 않는다.
- 작업은 단계별로 진행하고, 한 단계를 마치고 확인받은 뒤 다음으로 넘어간다.
- 파일만 만들고 끝내지 않고, 무엇을 왜 했는지 채팅으로도 설명한다.
- 시각화는 연구 끝에 한 번에 한다.

### 5.3 사용자 확인 없이 하지 않는 것

- 파일 생성(항상 **협의 → 사용자 컨펌 → 생성** 순서), 파일·디렉터리 삭제
- 기존 파일 30줄 이상 수정이나 덮어쓰기(수정 전에 바뀔 줄 수를 어림한다)
- 새 의존성(패키지, 라이브러리, CLI 도구) 추가
- 노션 쓰기 등 외부 기록
- **push는 매번 무조건 묻는다.** 이전에 받은 허락은 다음 push에 적용하지 않는다. force push, 브랜치 삭제, 태그 조작도 같다.
- `git add`, `git commit`은 허락과 관계없이 실행하지 않는다(9절).

요구가 두 가지 이상으로 해석되거나 정확한 답에 추가 정보가 필요하면 진행하지 않고 묻는다.

### 5.4 작업 종료 시 필수

작업이 끝나면 매번 `clave-record` 스킬 절차를 따른다. 생략하지 않는다.

1. `CLAUDE.local.md`의 현재 단계와 이 파일의 바뀐 규칙을 갱신한다.
2. 노션 연구 과정 기록과 설계서 반영은 `CLAUDE.local.md`의 노션 작업 방식이 있으면 그것을 따르고, 없으면 아래 절차대로 직접 기록한다(연구 과정 기록 추가, 설계가 바뀌었으면 설계서 해당 절과 16절 변경 이력 수정).
3. 커밋 시점이면 파일 목록과 메시지를 추천한다.

### 5.5 이 파일과 스킬 관리

- 이 파일은 **200줄 이하**로 유지한다. 원칙과 짧은 규칙만 두고, 세부 절차는 스킬로 옮긴다.
- 두 번 이상 반복되는 절차, 양식, 체크리스트는 `.claude/skills/<이름>/SKILL.md`로 스킬화하고, 이 파일에는 스킬 이름만 적는다. 스킬 생성도 파일 생성이므로 확인을 받는다.

| 스킬 | 언제 |
| --- | --- |
| `clave-record` | 작업 종료 기록(현재 단계, 노션 연구 과정 기록, 설계서 변경, 커밋 추천) |

## 6. 코드 규칙

- 코드 안의 모든 내용(식별자, 주석, docstring, 로그, 출력 메시지, 예외 메시지)은 **영어**로 작성한다.
- 주석은 코드와 같은 줄에 쓰지 않고, **반드시 윗줄에 따로** 쓴다.

```python
# Good: the comment sits on its own line
threshold = np.percentile(normal_scores, 95)

threshold = np.percentile(normal_scores, 95)  # Bad: inline comment
```

- 경로, 임계값, 모델 이름, 시드는 코드에 직접 쓰지 않고 `configs/*.yaml`에서 읽는다.
- 모든 무작위 요소는 설정 파일의 시드로 고정한다. VLM 디코딩은 greedy로 한다. 함수에는 타입 힌트를 붙인다.
- docstring과 키 이름에 설계서의 확인 항목 번호를 붙이지 않는다. 대응 관계는 연구 과정 기록에 남긴다.
- 스크립트는 프로젝트 루트 기준 상대 경로(`data/`, `results/`)를 쓴다. 코드와 결과 파일에 `/Users/...` 같은 절대 경로를 넣지 않는다.
- 같은 기능은 공용 모듈에 한 번만 구현한다. 다운로드 스크립트는 파일이 이미 있으면 건너뛴다.
- 지시문(prompt)은 버전을 붙여 고정하고, 바꾸면 새 버전 파일을 만든다.
- 생성 응답은 요청이 완전히 같을 때만 캐시에서 재사용한다. 외부 결과는 출처와 파일 해시를 함께 둔다.
- 비밀정보는 코드에 넣지 않고 `.env` 환경변수로 관리한다. `.env`를 쓰기 전에 `.gitignore` 등록을 확인하고, 항목 이름만 `.env.example`에 둔다.

### 6.1 결과 파일

- `results/*.json`에는 최소한 `run_id`, `git_commit`, `config_path`, `config_hash`, `dataset`, `seed`, `created_at`, `metrics`, `environment`(로컬, Kaggle 이미지 버전 등)를 넣는다.
- 결과를 만드는 스크립트는 **먼저 커밋한 뒤 실행**해, `git_commit`이 그 코드를 가리키게 한다.
- `results/` 파일은 손으로 고치지 않는다. 다시 계산해서 새 파일로 만든다.

## 7. 실행 환경

- 메모리 1GB 안팎이거나 몇 분 안에 끝나는 작업은 로컬에서 `caffeinate`·`nice`로 실행하고, 크고 오래 걸리는 작업과 GPU 작업만 Kaggle로 보낸다.
- 오래 걸리는 실행 전에는 반복을 줄인 축소판으로 저장까지 먼저 확인한다.
- Kaggle은 비공개 데이터셋·커널로 실행하고 결과를 내려받아 로컬에서 커밋한다. `kernel-metadata.json`처럼 계정 ID가 들어가는 파일은 스크립트로 만들고 추적하지 않는다. 데이터셋 새 버전은 처리가 끝난 뒤 커널을 실행하고, 실행 기록의 커밋을 확인한다.
- 승인된 의존성: torch, transformers(Qwen3.5는 5.2.0 이상), accelerate, bitsandbytes, pyyaml, pillow, torchvision(Kaggle 기본 설치본 사용, 별도 설치·버전 고정 안 함), kaggle CLI(로컬 `.venv`)
- 브라우저(Claude in Chrome)는 새 탭을 만들어 그 탭에서만 작업하고, 사용자가 열어 둔 탭에 접근하거나 닫지 않는다.
- macOS에서 터미널이 다운로드 폴더 권한이 없으면 `mv`가 `Operation not permitted`로 실패한다. Finder로 옮기거나 권한을 준다.

## 8. 규칙 고정

- 표본은 설정된 시드와 층화 규칙으로만 뽑는다. 원하는 방향으로 골라 뽑지 않는다.
- 확정 규칙은 노션 설계서 13절에 두고, 고정 시점(`configs/`와 프롬프트 파일 커밋)의 해시를 연구 과정 기록에 남긴다.

## 9. git

- **커밋은 사용자가 직접 한다.** Claude는 `git add`, `git commit`을 실행하지 않고, 커밋 시점에 파일 목록과 메시지만 추천한다(`clave-record` 4절).
- 단계 안의 작은 수정은 바로 커밋하지 않고 모아 두었다가, Kaggle에 올리는 등 실행 직전에 한 번 커밋한다(실행 한 번에 커밋 한 번). 커밋은 코드와 문서(`CLAUDE.md` 등)처럼 성격이 확실히 다를 때만 나누고, 같은 코드의 여러 수정은 한 커밋으로 합친다.
- 커밋 메시지는 Conventional Commits 형식의 영어로 쓰고, `Co-Authored-By`나 "Generated with Claude Code"를 넣지 않는다.
- 커밋 작성자는 저장소 로컬 설정의 GitHub noreply 주소다.
- pre-commit 훅 `.githooks/pre-commit`이 noreply가 아닌 작성자, 커밋 금지 파일, 로컬 사용자명·홈 경로·전역 git 이름과 이메일, 토큰 형태 문자열을 막는다. 클론마다 한 번 `git config core.hooksPath .githooks`로 켠다. 추가로 막을 개인 단어(Kaggle 사용자명, 노션 ID 등)는 `.git/info/forbidden-words`에 한 줄에 하나씩 적는다(추적 안 됨).
- push는 5.3절대로 매번 묻고, 허락받으면 `git push origin main` 또는 `git push origin stage/<단계>`만 쓴다.
- 기준 브랜치는 `main`이다. 작업은 단계마다 `stage/<단계>` 브랜치를 `main`에서 만들어 하고, 단계 브랜치는 한 번에 하나만 연다. 단계 목록은 `CLAUDE.local.md`에 있다.
- 단계를 마치면 결과가 실패여도 `git merge --no-ff`로 `main`에 병합한다. 병합 커밋이 단계 완료 표시다. 병합하지 않은 단계 브랜치는 지우지 않는다.
- 커밋은 rebase, amend, squash로 바꾸지 않고, 이미 만든 커밋은 합치지 않는다. 결과 파일과 연구 과정 기록이 그 해시를 가리킨다.
- 평소 이력은 `git log --first-parent main`으로 보고, 세부 이력은 단계 브랜치에 남긴다.
- 단계 브랜치도 원격에 올릴 수 있다. 병합한 뒤에도 원격 단계 브랜치는 지우지 않는다(지울 때는 허락을 받는다).
- 브랜치 생성, 병합, 삭제는 사용자가 하고, Claude는 명령만 추천한다.

## 10. 개인정보와 비밀정보

### 10.1 커밋 금지 대상

다음은 **절대 커밋하지 않고**, 공용 파일(이 파일, 코드, 설정, 노트북, 스킬, `docs/`, `results/`)에 옮겨 적지도 않는다.

- 개인정보: 이메일 주소, 실명, 학번·소속 계정, 로컬 사용자명이 드러나는 절대 경로, Kaggle 사용자명과 데이터셋·커널 ID, 개인 노션 페이지 URL·ID, 전화번호
- 비밀정보: `.env`, `kaggle.json`, `~/.kaggle/` 안의 토큰, API 키, 인증서
- 개인 설정: `CLAUDE.local.md`, `.claude/` 전체(설정, 스킬)
- 데이터: `data/`, `cache/`, 모델 가중치

개인 값은 `CLAUDE.local.md`나 `.env`에 두고 공용 파일에서는 위치만 가리킨다. 노트북은 커밋 전에 출력 셀에 로컬 경로나 계정명이 남지 않았는지 확인한다.

### 10.2 비밀정보 취급

- Claude는 `.env`, `.env.*`, `~/.kaggle/`, `kaggle.json`, `access_token` 같은 비밀 파일을 읽거나 출력하지 않는다. `.claude/settings.json`의 읽기 거부 규칙과 Bash 훅이 이를 강제한다. 훅에 막히면 우회하지 말고 사용자에게 실행을 부탁한다.
- 인증을 쓰는 CLI(kaggle, gh, gcloud, bq)는 출력과 오류를 보지 않고 종료 코드만 확인한다.
- `CLAUDE.local.md`에는 비밀 값이 아니라 위치만 적는다. 비밀이 노출되면 즉시 알리고 폐기·재발급을 권한다.

## 11. 금지 사항

- 데이터셋 이미지를 외부 API로 보내는 코드를 사용자 승인 없이 만들거나 실행하지 않는다(설계서 9.2절).
- 데이터셋 원본이나 캐시를 저장소나 공개 위치에 올리지 않는다. `data/`는 수정·덮어쓰기·커밋하지 않는다.
- `eval/` 밖에서 정답 마스크나 테스트 라벨을 읽지 않는다.
- 평가 지표를 본 뒤 고정 규칙을 조용히 바꾸지 않는다.
- 실행하지 않은 성능 수치를 문서·주석·보고서에 적지 않는다.
- 확인하지 않은 사실(라이선스, 데이터 수치, 선행 연구)을 확인된 것처럼 쓰지 않는다. 존재를 확인하지 않은 논문·출처·인용은 만들어내지 않고 "미확보"로 표기한다.
- 자료 조사 시 나무위키를 인용하지 않는다.
