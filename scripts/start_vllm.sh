#!/usr/bin/env bash
# Serve an open model on the AMD GPU with vLLM (run inside the AMD hackathon notebook terminal).
# The agent then talks to it at http://localhost:8000/v1 (see Option B in .env.example).
#
# Check vLLM is available first:   python -c "import vllm; print(vllm.__version__)"
# Check the GPU is visible:        rocm-smi

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"

vllm serve "$MODEL" \
  --port 8000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  > vllm.log 2>&1 &

echo "Starting vLLM with $MODEL (logs: vllm.log). Waiting for it to be ready..."
until curl -s http://localhost:8000/v1/models > /dev/null; do sleep 5; done
echo "vLLM is ready at http://localhost:8000/v1"
