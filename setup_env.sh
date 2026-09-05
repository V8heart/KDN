#!/usr/bin/env bash
# GridPulse 가상환경 세팅 (Ubuntu 22.04 / Python 3.10 / CUDA 13.0 / RTX 4090 x2)
# 사용: bash setup_env.sh  그다음  source .venv/bin/activate
set -e

echo "=== [1/4] venv 생성 ==="
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

echo "=== [2/4] 코어 의존성 설치 ==="
# 파이프라인 코어 (탐지/분석). 무거운 딥러닝 없이 동작하는 최소 세트.
pip install numpy pandas scipy scikit-learn

echo "=== [3/4] 텔레메트리 수집용 ==="
pip install nvidia-ml-py    # pynvml (NVML 폴링)

echo "=== [4/4] RAG + 경량 LLM용 ==="
# 임베딩 검색: faiss-cpu(가벼움) + sentence-transformers
pip install faiss-cpu sentence-transformers

# 로컬 LLM 서빙 옵션 (둘 중 택1, 나중에 교체 가능):
#  (A) Ollama 사용 시: 이 스크립트 밖에서 `curl -fsSL https://ollama.com/install.sh | sh`
#      후 `ollama pull qwen2.5:7b` 등. rag_analyzer.py는 기본적으로 Ollama HTTP를 호출.
#  (B) transformers 직접 사용 시 아래 주석 해제:
# pip install torch transformers accelerate

echo ""
echo "=== 완료 ==="
echo "GPU 확인:"
python -c "import pynvml; pynvml.nvmlInit(); print('GPU 수:', pynvml.nvmlDeviceGetCount()); pynvml.nvmlShutdown()" || echo "(pynvml 확인 실패 - GPU 없는 환경일 수 있음)"
echo ""
echo "다음: source .venv/bin/activate"
