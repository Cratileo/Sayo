import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["ORION_GMEM_CONTROL"] = 'v1'
# from load_model import QwenVLModel
import json
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, PreTrainedModel, AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration
import base64
import torch
from qwen_vl_utils import process_vision_info
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from types import MethodType
import pandas as pd
# import html
from typing import Type, Optional, Any, Tuple
from neo_func import NeoAttnforward
from qwen2_5_func import NeoforwardQwen2_5
import datasets
from datasets import Features, Value, load_dataset, Image
from trl.trainer.utils import entropy_from_logits

def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
    
def find_first_last_indices(tensor: torch.Tensor, value) -> tuple:
    if tensor.dim() != 1:
        raise ValueError("Input tensor must be 1-dimensional.")
    
    matches = (tensor == value).nonzero(as_tuple=True)[0]
    
    if matches.numel() == 0:
        return -1, -1
    else:
        return matches[0].item(), matches[-1].item()


def list_rfind(lst, value):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == value:
            return i
    return -1


def draw_plot(orig_img, attn, grid_w:int, grid_h:int, name:str):
    alpha = 0.4
    mask = attn.reshape(grid_h, grid_w).to(torch.float).cpu().numpy()
    vmin = np.percentile(mask, 1)
    vmax = np.percentile(mask, 99)

    mask_clipped = np.clip(mask, vmin, vmax)
    mask = (mask_clipped - vmin) / (vmax - vmin)

    # plt.imsave('./map/attn_map_test_origin.png', mask, cmap='jet', format='png')

    mask = cv2.resize(mask, (grid_w*28, grid_h*28))
    # plt.imsave('./map/attn_map_test.png', mask, cmap='jet', format='png', vmin=np.percentile(mask, 1), vmax=np.percentile(mask, 99))

    mask_uint8 = np.uint8(255 * mask)
    heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap, min(alpha, 1.0), orig_img, 1 - min(alpha, 1.0), 0)
    
    # plt.imshow(overlay)
    plt.axis('off')
    # plt.show()
    plt.imsave(f'./attn/visual/{name}.png', overlay)
    plt.clf()

#-----------------------------------------------------------------------------

class AttentionAnalyse:
    def __init__(self, model_name:str, save_dir:str, data_path:str, ModelClass:Type[PreTrainedModel], load_args:Optional[dict[str, Any]]=None, **kwargs):
        self.save_dir = save_dir
        self.data_path = data_path
        self.attention_maps = []
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.model = ModelClass.from_pretrained(
                    model_name,
                    dtype=torch.bfloat16,
                    attn_implementation="eager",
                    device_map="auto",
                    )

        self.processor = AutoProcessor.from_pretrained(model_name)

        # for name, _ in self.model.base_model.named_modules():
        #     if "attn" in name:
        #         print(name)

        module = self.model.model.language_model.layers[-1]
        # module.forward = MethodType(NeoBlockforward, module)
        # module.self_attn.forward = MethodType(NeoAttnforward, module.self_attn)
        module.self_attn.register_forward_hook(self.hook_decoder_cross_attention)
        print(f"Hooked layer Successfully")

    def hook_decoder_cross_attention(self, module, input, output):
        self.attention_maps.append(output[1].to('cpu'))

    def get_logits_and_entropy(self, inputs, completion_ids):
        prompt_ids = inputs.input_ids
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        completion_ids = completion_ids.unsqueeze(0)
        completion_mask = torch.ones_like(completion_ids, dtype=torch.long)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=-1)
        logits_to_keep = completion_ids.size(1)
        neo_inputs = {
            "input_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": inputs.get("image_grid_thw", None),
            "pixel_values": inputs.get("pixel_values", None),
        }
        all_entropies = []

        neo_inputs["logits_to_keep"] = logits_to_keep + 1

        neo_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        logits = self.model(**neo_inputs).logits
        # Exclude the last value: it corresponds to the next token pred
        logits = logits[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)

        with torch.no_grad():
            entropies = entropy_from_logits(logits)

        entropies = entropies[:,:-1].squeeze(0)

        return entropies


    def process(self, messages, idx, areas, answer:Tuple[str,str]=("","")):
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,return_dict=True,return_tensors="pt").to(self.model.device)

        image_id = find_first_last_indices(inputs['input_ids'].squeeze(0), 151655)
        # video_id = find_first_last_indices(inputs['input_ids'].squeeze(0), 151656)
        # audio_id = find_first_last_indices(inputs['input_ids'].squeeze(0), 151646)

        output = self.model.generate(**inputs, max_new_tokens=650, temperature=0.1, top_p=0.7, top_k=1, do_sample=True)

        generated_ids_trimmed = output[0][inputs.input_ids.shape[1]: ]
        words = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        generated_ids_trimmed_2 = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)]
        content = self.processor.batch_decode(generated_ids_trimmed_2, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
        entropy_per_token = self.get_logits_and_entropy(inputs, generated_ids_trimmed)
        #-------------------visual load---------------------
        # grid_w = int(inputs['image_grid_thw'][0][2].item() / 2)
        # grid_h = int(inputs['image_grid_thw'][0][1].item() / 2)

        # orig_img = cv2.imread('/gemini/space/yifq/zhaozy/ousiqu/frame_0000.png')
        # orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        # orig_img = cv2.resize(orig_img, (grid_w*28, grid_h*28))

        # alpha = 0.4
        # plt.figure(figsize=(10, 10))

        # shape b h q k
        text_attention = []
        image_attention = []
        area_attention = []
        
        text_eos = self.attention_maps[0].shape[3]

        for attn in self.attention_maps[1:]:
            avg_attn = attn.squeeze(0).mean(dim=0).squeeze(0)
            # attn_vision = avg_attn[-1, image_id[0]:image_id[1]+1]
            # draw_plot(orig_img, attn_vision, grid_w, grid_h, name=f"token_{cnt}")
            avg_text = avg_attn[image_id[1]+3:text_eos].mean()
            avg_image = avg_attn[image_id[0]:image_id[1]+1].mean()
            temp_list = []
            for a, b in areas:
                temp_list.append(avg_attn[image_id[0]+a:image_id[0]+b+1].mean().item())

            text_attention.append(avg_text.item())
            image_attention.append(avg_image.item())
            area_attention.append(np.mean(temp_list))

        df = pd.DataFrame({
            "token_id": generated_ids_trimmed[:len(entropy_per_token)].cpu().numpy().tolist(),
            "entropy": entropy_per_token.to(torch.float16).cpu().numpy().tolist(),
            "token": words[:len(entropy_per_token)],
            "to_text": text_attention[:len(entropy_per_token)],
            "to_image": image_attention[:len(entropy_per_token)],
            "to_area": area_attention[:len(entropy_per_token)],
            "answer": [answer[0]] + [None]*(len(entropy_per_token)-1),
            "short_answer": [answer[1]] + [None]*(len(entropy_per_token)-1),
            "completion": [content] + [None]*(len(entropy_per_token)-1),
        })

        df.to_parquet(f"{self.save_dir}/sample_{idx}.parquet", index=False)
        

    def run(self):
        features = Features({
            "image":Image(),
            "question": Value("string"), 
            "answer": Value("string"),
            "short_answer": Value("string"),
            'focus_area': datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
            "type": Value('string'),
            "id": Value('string'),
        })

        data = load_dataset("parquet", data_files=self.data_path, split='train', features=features)

        for idx, dt in tqdm(enumerate(data)):
            if os.path.exists(f"{self.save_dir}/{idx}.parquet"):
                continue
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": dt['image']
                        },
                        {"type": "text", "text": 'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.'},
                        {"type": "text", "text": dt['question']},
                    ],
                }
            ],
            try:
                self.process(messages, idx, dt['focus_area'], answer=(dt['answer'], dt['short_answer']))
            except torch.OutOfMemoryError:
                # print(f"OOM! Pass this case id: {idx}")
                with open(f'{self.save_dir}/error.log', 'a') as f:
                    f.write(f"[INFO] Pass {idx} by OOM\n")

            self.attention_maps = []

if __name__ == '__main__':
    model_name = '/gemini/space/yifq/zhaozy/ousiqu/attn/model_result/Qwen3-VL-8B-Instruct/merged_models/type9'
    data_path = '/gemini/space/yifq/zhaozy/ousiqu/attn/datasets/Analysis_0.5k.parquet'
    aa = AttentionAnalyse(
        model_name,
        save_dir='/gemini/space/yifq/zhaozy/ousiqu/attn/visual/Type9',
        data_path=data_path,
        ModelClass=Qwen3VLForConditionalGeneration,
    )

    aa.run()

