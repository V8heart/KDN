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

# (선택) 로컬 LLM: Ollama 설치 후
#   curl -fsSL https://ollama.com/install.sh | sh
#   ollama pull qwen2.5:7b     # 나중에 더 큰 모델로 교체 가능

# 파이프라인 실행 (수집된 텔레메트리 CSV 대상)
python pipeline/run_pipeline.py \
    --telemetry data.csv \
    --baseline-mean 120 \
    --rag-backend sbert \
    --llm-backend ollama --llm-model qwen2.5:7b
```

## 구성요소

| 경로 | 역할 |
|---|---|
| `corpus/*.md` | 공격 시그니처 지식베이스. **새 공격은 여기 문서만 추가하면 대응** |
| `pipeline/features.py` | 1·2단계: 텔레메트리 → 통계 피처 → 정성적 자연어 설명 |
| `pipeline/rag_analyzer.py` | 3단계: RAG 검색(sbert/tfidf) + LLM 판정(ollama/stub) |
| `pipeline/run_pipeline.py` | 전체 오케스트레이션 |
| `bit2watt_impl/` | (별도) SWMA/LTMA 공격 재현 + DCGM 수집 — 검증셋 구축용 |

## 백엔드 옵션 (점진적 구축용)

- **RAG**: `sbert`(권장, 의미 임베딩) / `tfidf`(오프라인 폴백, 모델 다운로드 불필요)
- **LLM**: `ollama`(로컬 경량 모델) / `stub`(LLM 없이 규칙 기반, 파이프라인 검증용)

→ LLM 없이 `--rag-backend tfidf --llm-backend stub`으로 먼저 파이프라인 전체를
   검증하고, 그다음 sbert + ollama로 성능을 올리는 순서를 권장.

## 검증 상태

- ✅ features.py: 규칙적 사각파(SWMA류) vs 평평한 고부하(크립토재킹류)를 정성적으로 정확히 구분
- ✅ rag_analyzer.py: SWMA류 설명 → `swma` 문서 1순위 매칭 (tfidf 기준)
- ✅ run_pipeline.py: 정상+SWMA 혼합 합성 텔레메트리에서 공격 구간만 정확히 선별 후 swma 매칭
- (실서버) sbert 임베딩 + ollama LLM로 교체 시 매칭 정확도·설명 품질 향상 예상

## 확장 (제품 비전)

- `corpus/`에 문서 추가만으로 신종 위협 대응 (재학습 불필요) — MA-IDS의 "경험 라이브러리" 패턴
- 커넥터 아키텍처로 향후 운영 매뉴얼/위협 인텔리전스/외부 데이터 소스를 RAG 컨텍스트에 연결 가능
