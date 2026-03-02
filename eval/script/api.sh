
EVAL_DIR="ABSOLUTE/PATH/THIS/DIR"
DATA_DIR="ABSOLUTE/PATH/DATASETS/DIR"
RESULTS_DIR="ABSOLUTE/PATH/RESULTS/DIR"
mkdir -p "$RESULTS_DIR"

cd "$EVAL_DIR"

python main.py \
  --model "gpt4o" \
  --output-dir "$RESULTS_DIR" \
  --data-path "$DATA_DIR" \
  --datasets ChartQA,mmstar,MMErealworld, \
  --max-model-len 10240 \
  --temperature 0.1 \
  --top-p 0.1 \
  --version "grpo" \
  --url "" \
  --api-key "" \
  
echo "Finished, close server (PID: $VLLM_PID)..."
kill $VLLM_PID