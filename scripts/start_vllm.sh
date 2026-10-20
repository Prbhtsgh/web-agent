#!/usr/bin/env bash
# Serve an open model on the AMD GPU with vLLM. Run it in the AMD notebook terminal, from the repo folder:
#
#   bash scripts/start_vllm.sh                              # picks the largest model that fits the GPU
#   bash scripts/start_vllm.sh Qwen/Qwen2.5-14B-Instruct    # or name one yourself
#
# Model choice (Qwen2.5 family: the prompts were tuned on qwen2.5:7b, and vLLM on ROCm supports it well):
#   GPU memory >= 120 GB (e.g. MI300X, 192 GB)  -> Qwen/Qwen2.5-32B-Instruct  (~65 GB of weights)
#   GPU memory >=  40 GB (e.g. W7900D, 48 GB)   -> Qwen/Qwen2.5-14B-Instruct  (~30 GB of weights)
#   otherwise                                   -> Qwen/Qwen2.5-7B-Instruct   (~15 GB of weights)
#
# Weights are downloaded to $HF_HOME (default ~/.cache/huggingface). They are bigger than the ~25 GB persistent
# quota, so point HF_HOME at a large non-persistent disk (see README) and accept a re-download per session.
set -euo pipefail

PORT="${PORT:-8000}"

vram_gb=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 2**30) if torch.cuda.is_available() else 0)")
if [ "$vram_gb" -eq 0 ]; then
  echo "No GPU visible to PyTorch. Check with: rocm-smi" >&2
  exit 1
fi

if [ $# -ge 1 ]; then
  MODEL="$1"
elif [ "$vram_gb" -ge 120 ]; then
  MODEL="Qwen/Qwen2.5-32B-Instruct"
elif [ "$vram_gb" -ge 40 ]; then
  MODEL="Qwen/Qwen2.5-14B-Instruct"
else
  MODEL="Qwen/Qwen2.5-7B-Instruct"
fi
case "$MODEL" in
  *32B*) need_gb=70 ;;
  *14B*) need_gb=35 ;;
  *) need_gb=20 ;;
esac

# Enough disk for the weights? A half-finished download fails late and wastes GPU quota.
cache_dir="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$cache_dir"
free_gb=$(python -c "import shutil; print(shutil.disk_usage('$cache_dir').free // 2**30)")
echo "GPU memory: ${vram_gb} GB | model: $MODEL | download cache: $cache_dir (${free_gb} GB free, need ~${need_gb} GB)"
if [ "$free_gb" -lt "$need_gb" ] && [ ! -d "$cache_dir/hub/models--${MODEL//\//--}" ]; then
  echo "Not enough free disk in $cache_dir. Set HF_HOME to a bigger disk (see 'df -h'), or pick a smaller model." >&2
  exit 1
fi

vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  > vllm.log 2>&1 &
pid=$!

echo "Starting vLLM (pid $pid, logs: vllm.log). Downloading and loading can take 5-15 minutes..."
until curl -s "http://localhost:$PORT/v1/models" > /dev/null; do
  if ! kill -0 "$pid" 2> /dev/null; then
    echo "vLLM stopped. Last lines of vllm.log:" >&2
    tail -n 30 vllm.log >&2
    exit 1
  fi
  sleep 10
done

echo "vLLM is ready. Point the agent at it with:"
echo "  export LLM_BASE_URL=http://localhost:$PORT/v1 LLM_API_KEY=EMPTY LLM_MODEL=$MODEL"
