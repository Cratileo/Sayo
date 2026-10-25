EVAL_DIR="ABSOLUTE/PATH/THIS/DIR"
DATA_DIR="ABSOLUTE/PATH/DATASETS/DIR"
RESULTS_DIR="ABSOLUTE/PATH/RESULTS/DIR"
mkdir -p "$RESULTS_DIR"

cd "$EVAL_DIR"

export CUDA_VISIBLE_DEVICES=4,5,6,7

# CharXiv,m3cot,mathvision,MMErealworld,wemath,ai2d,vstar,ChartQA,mmstar

python main.py \
  --model YOUR/MODEL/PATH \
  --output-dir "$RESULTS_DIR" \
  --data-path "$DATA_DIR" \
  --datasets RealWorldQA,hallubench \
  --max-model-len 31240 \
  --temperature 0.1 \
  --version "grpo" \
  --top-p 0.1 \

python score.py \
  --folder_path "$RESULTS_DIR"