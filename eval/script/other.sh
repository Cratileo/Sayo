# export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_USE_V1=0

GPU_ID="0,1"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export ORION_GMEM_CONTROL=v1
PORT=10721
# export HF_ENDPOINT=https://hf-mirror.com

MODEL_NAME="R1-Onevision-7B"

echo "Start vLLM Server..."
vllm serve /gemini/space/yifq/zhaozy/models/$MODEL_NAME \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --max-parallel-loading-workers 2 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --port $PORT > vllm_server_01.log 2>&1 &

VLLM_PID=$!

echo "Waiting vLLM Load (Port $PORT)..."
while ! curl -s http://localhost:$PORT/v1/models > /dev/null; do
    echo "Loading..."
    sleep 20
done
echo "Done"

EVAL_DIR="/gemini/space/yifq/zhaozy/ousiqu/attn/eval/other"
DATA_DIR="/gemini/space/yifq/zhaozy/ousiqu/attn/datasets"
RESULTS_DIR="/gemini/space/yifq/zhaozy/ousiqu/attn/results/new/$MODEL_NAME"
mkdir -p "$RESULTS_DIR"

cd "$EVAL_DIR"

# CharXiv,hallubench,m3cot,mathverse,mathvision,mathvista,MMErealworld,TableVQA,wemath,ai2d,vstar,RealWorldQA,ChartQA

python main.py \
  --model "$MODEL_NAME" \
  --output-dir "$RESULTS_DIR" \
  --data-path "$DATA_DIR" \
  --datasets m3cot,mathvision,MMErealworld,wemath,ai2d,vstar,ChartQA,mmstar \
  --max-model-len 10240 \
  --temperature 0.1 \
  --top-p 0.1 \
  --version "grpo" \
  --port $PORT
  
echo "Finished, close server (PID: $VLLM_PID)..."
kill $VLLM_PID
