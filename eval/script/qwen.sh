
GPU_ID="0,1"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export ORION_GMEM_CONTROL=v1
PORT=10721
# export HF_ENDPOINT=https://hf-mirror.com

echo "Start vLLM Server..."
vllm serve YOUR/MODEL/PATH \
    --served-model-name Qwen3-VL-8B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 25000 \
    --max-parallel-loading-workers 2 \
    --gpu-memory-utilization 0.95 \
    --port $PORT > vllm_server.log 2>&1 &

VLLM_PID=$!

echo "Waiting vLLM Load (Port $PORT)..."
while ! curl -s http://localhost:$PORT/v1/models > /dev/null; do
    echo "Loading..."
    sleep 20
done
echo "Done"

EVAL_DIR="ABSOLUTE/PATH/THIS/DIR"
DATA_DIR="ABSOLUTE/PATH/DATASETS/DIR"
RESULTS_DIR="ABSOLUTE/PATH/RESULTS/DIR"
mkdir -p "$RESULTS_DIR"

cd "$EVAL_DIR"

# CharXiv,m3cot,mathvision,MMErealworld,wemath,ai2d,vstar,ChartQA,mmstar

python main.py \
  --model "Qwen3-VL-8B-Instruct" \
  --output-dir "$RESULTS_DIR" \
  --data-path "$DATA_DIR" \
  --datasets MMErealworld \
  --max-model-len 10240 \
  --temperature 0.1 \
  --version "grpo" \
  --top-p 0.1 \
  --port $PORT
  
echo "Finished, close server (PID: $VLLM_PID)..."
kill $VLLM_PID