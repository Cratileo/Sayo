import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ORION_GMEM_CONTROL"] = 'v1'
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

base_model_path = "PATH/YOUR/BASE/MODEL"
lora_path = "PATH/YOUR/LORA/CHECKPOINT"
output_path = "PATH/YOUR/OUTPUT/MODEL"

# load base
model = AutoModelForImageTextToText.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)

# load lora
model = PeftModel.from_pretrained(model, lora_path)

# merge
model = model.merge_and_unload()

# save
model.save_pretrained(output_path, safe_serialization=True)
processor = AutoProcessor.from_pretrained(base_model_path)
processor.save_pretrained(output_path)