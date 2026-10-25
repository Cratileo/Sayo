from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import os

class VLLMLocalPipeline():
    def __init__(self, **model_args):
        self.llm = LLM(**model_args)
        self.processor = AutoProcessor.from_pretrained(model_args['model'])
        self.sampling_params = SamplingParams(
            temperature=0.1, max_tokens=2048, stop_token_ids=[], top_p=0.1
        )

    def generate(self, messages: list[dict]) -> str:
        inputs = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_data, _ = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
        )
        llm_inputs = {"prompt": inputs}
        if image_data:
            llm_inputs["multi_modal_data"] = {"image": image_data}

        outputs = self.llm.generate(
            [llm_inputs], sampling_params=self.sampling_params, use_tqdm=False
        )
        response = ""
        if isinstance(outputs, list):
            for o in outputs:
                if o.outputs[0].text != "":
                    response = o.outputs[0].text
        else:
            response = outputs.outputs[0].text
        return response