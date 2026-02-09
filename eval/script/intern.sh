# export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_USE_V1=0

GPU_ID="0,1"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export ORION_GMEM_CONTROL=v1
PORT=10721
# export HF_ENDPOINT=https://hf-mirror.com

echo "Start vLLM Server..."
vllm serve /gemini/space/yifq/zhaozy/ousiqu/attn/model_result/InternVL3_5-8B/merged_models/type3_custom \
    --served-model-name InternVL3_5-8B \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 25000 \
    --max-parallel-loading-workers 2 \
    --gpu-memory-utilization 0.95 \
    --trust_remote_code \
    --port $PORT > vllm_server_02.log 2>&1 &

VLLM_PID=$!

echo "Waiting vLLM Load (Port $PORT)..."
while ! curl -s http://localhost:$PORT/v1/models > /dev/null; do
    echo "Loading..."
    sleep 20
done
echo "Done"

EVAL_DIR="/gemini/space/yifq/zhaozy/ousiqu/attn/eval/other"
DATA_DIR="/gemini/space/yifq/zhaozy/ousiqu/attn/datasets"
RESULTS_DIR="/gemini/space/yifq/zhaozy/ousiqu/attn/results/internvl/new/type3"
mkdir -p "$RESULTS_DIR"

cd "$EVAL_DIR"

python main.py \
  --model "InternVL3_5-8B" \
  --output-dir "$RESULTS_DIR" \
  --data-path "$DATA_DIR" \
  --datasets mmstar \
  --max-model-len 10240 \
  --temperature 0.1 \
  --top-p 0.1 \
  --version "grpo" \
  --port $PORT
  
echo "Finished, close server (PID: $VLLM_PID)..."
kill $VLLM_PID