import os
import math
from PIL import Image
from typing import List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from mathruler.grader import extract_boxed_content, grade_answer
import io
import base64

def load_image(image_path: str | Image.Image, min_pixels: int, max_pixels: int) -> Image.Image:
    """Load and preprocess an image"""
    try:
        if isinstance(image_path, Image.Image):
            image = image_path
        else:
            image = Image.open(image_path).convert("RGB")
        
        # Resize if too large or too small
        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def prepare_prompts(dataset_name: str, samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    """Prepare prompts for all samples"""
    prompts = []
    metadata = []
    
    for item in tqdm(samples, desc=f"Preparing {dataset_name} prompts"):
        # Skip if image doesn't exist
        if not os.path.exists(item["image_path"]):
            continue
        
        # Load image
        image = load_image(item["image_path"], args.min_pixels, args.max_pixels)
        if image is None:
            continue
        
        # if "<image>" not in item['question']:
        #     item['question'] = "<image>\n" + item['question']

        # Create prompt
        if args.version == "grpo":
            prompt_text = f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        elif args.version == "back":
            prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question']} {args.system_prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif args.version == "hint":
            prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> {args.system_prompt} \n Qwestion: {item['question']}<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise
        
        prompts.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        })
        
        metadata.append({
            "dataset": dataset_name,
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "prompt": prompt_text,
            **{k: v for k, v in item.items() if k not in ["image_path", "dataset", "id", "question", "answer"]}
        })
    
    return prompts, metadata

def process_outputs_simplified(outputs, metadata) -> List[Dict]:
    results = []
    for i, output in enumerate(outputs):
        prediction = output.strip()
        meta = metadata[i]
        
        result = {
            "id": meta["id"],
            "question": meta["question"],
            "answer": meta["answer"],
            "prediction": prediction
        }
        results.append(result)
    
    return results


def encode_image_to_base64(image: str|Any) -> str:
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # Handle PIL Image object
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded_string = base64.b64encode(buffer.read()).decode('utf-8')
    return encoded_string 


def prepare_messages(dataset_name: str, samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    """Prepare prompts for all samples"""
    messages = []
    metadata = []
    
    for item in tqdm(samples, desc=f"Preparing {dataset_name} prompts"):
        # Skip if image doesn't exist
        if isinstance(item['image_path'], str) and not os.path.exists(item["image_path"]):
            continue
        
        # Load image
        image = load_image(item["image_path"], args.min_pixels, args.max_pixels)
        if image is None:
            continue
        
        # if "<image>" not in item['question']:
        #     item['question'] = "<image>\n" + item['question']
        image_url = encode_image_to_base64(image)

        message = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": args.system_prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {'url': "data:image/png;base64," + image_url},
                    },
                    {"type": "text", "text": f"{item['question']}"},
                ],
            }
        ]

        # Create prompt
        # if args.version == "grpo":
        #     prompt_text = f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        # elif args.version == "back":
        #     prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question']} {args.system_prompt}<|im_end|>\n<|im_start|>assistant\n"
        # elif args.version == "hint":
        #     prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> {args.system_prompt} \n Qwestion: {item['question']}<|im_end|>\n<|im_start|>assistant\n"
        # else:
        #     raise
        
        # prompts.append({
        #     "prompt": prompt_text,
        #     "multi_modal_data": {"image": image},
        # })

        messages.append(message)
        
        
        metadata.append({
            "dataset": dataset_name,
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "prompt": messages,
            **{k: v for k, v in item.items() if k not in ["image_path", "dataset", "id", "question", "answer"]}
        })
    
    return messages, metadata