# HF_MODEL="/absolute/path/to/LooK-Back/MLLMs/Solution-back-7B"
# HF_MODEL="/gemini/space/yifq/zhaozy/models/InternVl3_5-8B"
EVAL_DIR="/gemini/space/yifq/zhaozy/ousiqu/attn/eval/other"
DATA_DIR="/gemini/space/yifq/zhaozy/ousiqu/attn/datasets"
RESULTS_DIR="/gemini/space/yifq/zhaozy/ousiqu/attn/results/new/gpt-4o"
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
  --url "https://research-01-01.openai.azure.com/" \
  --api-key "" \
  
echo "Finished, close server (PID: $VLLM_PID)..."
kill $VLLM_PID