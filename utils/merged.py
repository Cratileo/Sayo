import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ORION_GMEM_CONTROL"] = 'v1'
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

base_model_path = "/gemini/space/yifq/zhaozy/models/InternVL3_5-8B-HF"
lora_path = "/gemini/space/yifq/zhaozy/ousiqu/attn/model_result/InternVL3_5-8B/2nd/checkpoint-2475"
output_path = "/gemini/space/yifq/zhaozy/ousiqu/attn/model_result/InternVL3_5-8B/merged_models/type2_2475"

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