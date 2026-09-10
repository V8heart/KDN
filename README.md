# GridPulse — GPU 워크로드 이상탐지 파이프라인

새 공격이 나올 때마다 탐지 모델을 재학습하는 대신, **범용 이상 필터로 의심 활동을
선별하고, 알려진 공격 시그니처 문서와 RAG로 대조**하는 실용 파이프라인.

## 설계 근거 (검증된 2-tier IDS 패턴)

이 구조는 침입탐지(IDS) 분야에서 확립된 정석입니다:
- **LLM-Enhanced Two-Tier IDS** (Springer 2026): 가벼운 통계 모니터링으로 상시 감시하다가,
  이상 감지 시에만 무거운 LLM 분석을 on-demand로 촉발 → 우리의 "의심될 때만 고해상도 분석"과 동일.
- **POSEIDON 2-tier IDS**: 이상탐지(1단계) + 시그니처매칭(2단계) 결합.
- 이 패턴은 "하루 1,000개 알람을 40개, 다시 5개로 압축"하는 것으로 실증됨.

**우리가 하지 않는 것**: Bit2Watt의 kHz 원신호를 직접 탐지 (물리 계측 영역, 우선순위 제외).
**우리가 하는 것**: 관측 가능한 해상도에서 의심 활동을 선별 → 알려진 공격과 대조 → 설명 생성.

## 파이프라인 3단계

```
DCGM/NVML 텔레메트리
   ↓
[1단계] 상시 저비용 스크리닝 — 윈도우별 변동폭 z-score, 의심 후보만 선별
   ↓ (후보만)
[2단계] 고해상도 특성 분석 — 주기성/duty 규칙성/평균유지 여부를 정성적 설명으로 변환
   ↓
[3단계] RAG 대조 + LLM 판정 — 알려진 공격 시그니처와 의미 기반 매칭, 위험도+근거 생성
```

## 서버 요구사항 (검증된 환경)

- Ubuntu 22.04, Python 3.10+, CUDA 13.0, RTX 4090 (실측 환경: 4090 x2, 48코어, 125GB RAM)
- GPU 한 장으로 공격/워크로드 재현, 다른 한 장으로 로컬 LLM 서빙 가능

## 설치 및 실행

```bash
bash setup_env.sh
source .venv/bin/activate

# (선택) 로컬 LLM: 이 서버에는 Ollama 0.32.1이 이미 설치되어 있음
#   ollama serve   # 미실행 시
#   검증된 모델: gemma3:12b  (qwen3:14b는 format=json에서 빈 응답이 나와 비권장)

# 파이프라인 실행 (수집된 텔레메트리 CSV 대상)
python pipeline/run_pipeline.py \
    --telemetry dataset/synthetic/all_v2.csv \
    --baseline-mean 120 \
    --rag-backend tfidf \
    --llm-backend ollama \
    --llm-model gemma3:12b
```

오프라인/빠른 검증만 할 때는 `--llm-backend stub`을 쓰면 됩니다.
### 합성 데이터셋과 평가

```bash
python dataset/build_synthetic.py
python pipeline/eval_retrieval.py
python pipeline/eval_detection.py
pytest -q
```

합성 데이터는 `session_id + gpu_id` 단위로 처리하므로 서로 다른 클래스의
경계가 하나의 분석 윈도우에 섞이지 않는다.

### 실측 텔레메트리

```bash
python -m bit2watt_impl.collect_telemetry \
  --output dataset/real/observability/idle-100ms.csv \
  --duration 30 --interval-ms 100 --gpu-ids 0 \
  --label normal_baseline --job-type normal_baseline

python dataset/observability.py \
  dataset/real/observability/idle-100ms.csv
```

`--interval-ms`는 API 호출 요청 주기이지 센서의 물리 측정률이 아니다.
`actual_interval_ms`, `value_changed`와 observability summary로 실효 갱신률을
별도로 확인해야 한다.

### 통제된 실측 워크로드

다른 사용자의 GPU 작업이 없는지 먼저 확인한 다음에만 실행한다.

```bash
python dataset/run_capture.py --workload baseline --duration 30 \
  --confirm-shared-gpu-safe
python dataset/run_capture.py --workload swma --duration 30 \
  --period 1 --duty-cycle 0.5 --confirm-shared-gpu-safe
```

SWMA 코드는 Bit2Watt persistent kernel의 동일 재현이 아니라, PyTorch CUDA
연산과 sleep으로 소프트웨어 telemetry에서 보이는 규칙적 변동을 만드는 근사다.
2-GPU 캡처는 `distributed` 또는 `swma_multi` workload를 사용한다.

## 구성요소

| 경로 | 역할 |
|---|---|
| `corpus/*.md` | 공격 시그니처 지식베이스. **새 공격은 여기 문서만 추가하면 대응** |
| `pipeline/features.py` | 1·2단계: 텔레메트리 → 통계 피처 → 정성적 자연어 설명 |
| `pipeline/rag_analyzer.py` | 3단계: RAG 검색(sbert/tfidf) + LLM 판정(ollama/stub) |
| `pipeline/run_pipeline.py` | 전체 오케스트레이션 |
| `pipeline/physics_correlation.py` | Cyber 후보와 Physics Tier B 결과를 `attack_id`로 결합 |
| `dataset/` | 공통 스키마, 합성 데이터, 캡처·평가 도구 |
| `bit2watt_impl/physics/` | 공개 Kundur/WECC 계통에서의 동적 응답 검증 |
| `bit2watt_impl/` | NVML 수집과 SWMA/LTMA-like 검증 워크로드 |
| `workloads/` | 정상 PyTorch hard-negative 워크로드 |

## 백엔드 옵션 (점진적 구축용)

- **RAG**: `sbert`(권장, 의미 임베딩) / `tfidf`(오프라인 폴백, 모델 다운로드 불필요)
- **LLM**: `ollama`(로컬 경량 모델) / `stub`(LLM 없이 규칙 기반, 파이프라인 검증용)

→ LLM 없이 `--rag-backend tfidf --llm-backend stub`으로 먼저 파이프라인 전체를
   검증하고, 그다음 sbert + ollama로 성능을 올리는 순서를 권장.

## 검증 상태

- ✅ features.py: 규칙적 사각파(SWMA류) vs 평평한 고부하(크립토재킹류)를 정성적으로 정확히 구분
- ✅ rag_analyzer.py: SWMA류 설명 → `swma` 문서 1순위 매칭 (tfidf 기준)
- ✅ 합성 3개 공격 + 6개 정상 세션을 세션 경계 없이 생성
- ✅ Stage1 다중 필터: 합성 공격 윈도우 후보 recall 1.0
- ✅ TF-IDF retrieval top-1 및 정상 hard-negative 기각률 1.0 (합성셋 기준)
- ✅ NVML idle smoke capture와 observability summary 검증
- ✅ Ollama `gemma3:12b`로 실측 SWMA multi-GPU 후보에 대해 설명형 판정 검증 (`_backend: ollama:gemma3:12b`)
- (선택) `--rag-backend sbert`로 임베딩 검색 품질 추가 향상 가능
- ✅ Physics Tier B Phase 1: Kundur 7개 프로필 중 6개 수렴, 30% burst는
  `converged=false`로 기록
- ✅ Kundur 스윕: 28개 중 26개 수렴. 전역 `osc_std` peak는 0.1 Hz로,
  약 0.6 Hz 전역 봉우리는 확인되지 않음
- ✅ WECC 179 GENCLS 스윕: 28/28 수렴. 전역 peak는 1.05 Hz로,
  별도 공식 gallery 사례의 0.37 Hz 전역 봉우리를 재현하지 못함
- ⚠ Cyber–physics join의 사용 가능 공격군은 3개뿐이므로 상관계수는 기술통계이며,
  PTPS 수치 가중치는 확정하지 않음

### RTX 4090 제약

- `DCGM_FI_PROF_*`, NVLink, MIG는 사용하지 않는다.
- 메모리 사용량은 NVML을 우선 사용한다.
- 시스템 nvcc 11.5는 Ada `sm_89` 커널 컴파일에 부적합하므로, 실측 workload는
  `.venv`의 CUDA 지원 PyTorch를 사용한다.
- `dataset/real/`은 용량과 개인정보 가능성 때문에 Git에서 제외된다.

## Physics Tier B — 공개 동적 테스트 계통 검증

Cyber Stage 1은 GPU 텔레메트리 후보 선별이고, Physics Tier B는 별도의
오프라인 시뮬레이션이다. Tier B는 고위험으로 분류된 상대 부하 파형이 공개
Kundur/WECC 테스트 계통에서 더 큰 발전기 속도 응답을 만드는지 확인한다.
GPU 전력 W를 실제 계통 MW로 직접 환산하지 않는다.
`attack_id`, 파형 종류와 주파수는 양쪽에서 공유하지만, `cyber_amplitude_frac`은
관측된 GPU 파형을 설명하고 `amplitude_frac`은 공개 테스트계통에 사전 등록한
민감도 주입값이다. 둘 사이를 실제 설비 환산 관계로 해석하면 안 된다.

ANDES 2.0은 Python 3.11 이상이 필요하므로 기존 `.venv`와 분리한다.

```bash
./setup_physics_env.sh
source .venv-physics/bin/activate

# Phase 1: Kundur 정상/주기/버스트/램프 및 GridPulse 공격 프로필
python -m bit2watt_impl.physics.stage1_kundur

# Phase 2: Kundur 0.1~1.45 Hz 스윕
python -m bit2watt_impl.physics.frequency_sweep

# Phase 3: WECC smoke 후 주파수 스윕
python -m bit2watt_impl.physics.stage3_wecc
```

Cyber 결과와 결합:

```bash
source .venv/bin/activate
python dataset/build_synthetic.py
python pipeline/run_pipeline.py \
  --telemetry dataset/synthetic/all_v2.csv --baseline-mean 120 \
  --rag-backend tfidf --llm-backend stub \
  --out dataset/eval/cyber_pipeline_results.json

python pipeline/physics_correlation.py \
  --cyber-json dataset/eval/cyber_pipeline_results.json \
  --physics-csv dataset/eval/kundur_attack_features.csv \
  --out dataset/eval/cyber_physics_joined.csv

source .venv-physics/bin/activate
python pipeline/eval_physics_correlation.py
```

Tier B 결과의 `anomaly_score`는 검증된 물리 위험도가 아니라 Cyber Stage 1의
투명한 0~1 스크리닝 휴리스틱이다. 표본이 부족하면 PTPS 수치 가중치를 확정하지
않고 방향성 제안만 저장한다. 수렴 실패는 누락하지 않고 `converged=false`와
`failure_reason`으로 남긴다.

### 해석 제한

- 결과는 반드시 “공개 동적 테스트 계통(Kundur/WECC)에서”로 한정한다.
- 한국 실제 전력망의 주파수 하락이나 피해 규모로 일반화하지 않는다.
- KPG-193 Tier A(정적·위치별 민감도)와 Tier B(동적·범용 테스트 계통)는 별개다.
- Kundur 내장 케이스의 기존 t=2초 선로 Toggle은 실험 파형과 원인을 분리하기 위해
  비활성화하며, 이 사실을 결과 CSV에 기록한다.

## 확장 (제품 비전)

- `corpus/`에 문서 추가만으로 신종 위협 대응 (재학습 불필요) — MA-IDS의 "경험 라이브러리" 패턴
- 커넥터 아키텍처로 향후 운영 매뉴얼/위협 인텔리전스/외부 데이터 소스를 RAG 컨텍스트에 연결 가능
