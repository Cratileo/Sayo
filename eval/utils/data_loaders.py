import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from datasets import load_dataset

def load_wemath_dataset(data_path: str) -> List[Dict]:
    """Load WeMath dataset"""
    # image_root = os.path.join(data_path, "We-Math/__MACOSX/data")
    # data_path = os.path.join(data_path, "We-Math/testmini.json")
    # with open(data_path, "r", encoding="utf-8") as f:
    #     data = json.load(f)
    data = load_dataset("/gemini/space/yifq/zhaozy/ousiqu/attn/datasets/We-Math", split="testmini")
    
    dataset = []
    for item in data:
        # Determine the image path
        # image_path = os.path.join(image_root, item["image_path"])

        dataset.append({
            "id": item["ID"] + "@" + item["key"],
            "image_path": item['image_path'],
            "question": f"{item['question']}\n\nOptions: {item['option']}\nYour final answer must be one of A, B, C, or D.",
            "answer": item["answer"],
            "dataset": "wemath"
        })
    
    return dataset

def load_mathvista_dataset(data_path: str) -> List[Dict]:
    """Load MathVista dataset"""
    image_base_dir = os.path.join(data_path, "MathVista/images")
    dataset_raw = load_dataset("/gemini/space/yifq/zhaozy/ousiqu/attn/datasets/MathVista", split="testmini")
    
    dataset = []
    mapping = {
        "0": "A", "1": "B", "2": "C", "3": "D",
        "4": "E", "5": "F", "6": "G", "7": "H"
    }
    
    for item in dataset_raw:
        if item["question_type"] == "multi_choice":
            idx = item["choices"].index(item["answer"])
            answer = mapping[str(idx)]
        else:
            answer = item["answer"]
        
        dataset.append({
            "id": item.get("pid", ""),
            # "image_path": os.path.join(image_base_dir, item["image"]),
            'image_path': item['decoded_image'],
            "question": item["query"],
            "answer": answer,
            "task": item["metadata"]["task"],
            "dataset": "mathvista"
        })
    
    return dataset

def load_mathverse_dataset(data_path: str) -> List[Dict]:
    """Load MathVerse dataset"""
    image_base_dir = os.path.join(data_path, "MathVerse/images")
    data_path = os.path.join(data_path, "MathVerse/testmini.json")
    # data_path = os.path.join(data_path, "mathverse/testmini-test.json")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        dataset.append({
            "id": item.get("sample_index", ""),
            "image_path": os.path.join(image_base_dir, item["image"]),
            "question": item["query_cot"],
            "question_for_eval": item["question_for_eval"],
            "answer": item["answer"],
            "problem_version": item["problem_version"],
            "dataset": "mathverse"
        })
    
    return dataset

def load_GeoMath_dataset(data_path: str) -> List[Dict]:
    json_path = os.path.join(data_path, "GeoMath/GeoMath-data.json")
    geomath_base_path = os.path.join(data_path, "GeoMath")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        item_id = item.get("id", "")
        images = item.get("images", [])
        full_image_path = os.path.join(geomath_base_path, images[0])
        messages = item.get("messages", [])
        user_content = ""
        assistant_content = ""
        
        for message in messages:
            if message.get("role") == "user":
                user_content = message.get("content", "")
            elif message.get("role") == "assistant":
                assistant_content = message.get("content", "")
        
        dataset.append({
            "id": item_id,
            "image_path": full_image_path,
            "question": user_content,
            "question_for_eval": user_content,
            "answer": assistant_content,
            "problem_version": 'GeoMath',
            "dataset": "GeoMath"
        })
    
    return dataset

def load_Tallyqa_dataset(data_path: str) -> List[Dict]:
    json_path = os.path.join(data_path, "Tallyqa/Tallyqa.json")
    Tallyqa_base_path = os.path.join(data_path, "Tallyqa")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        item_id = item.get("id", "")
        images = item.get("images", [])
        full_image_path = os.path.join(Tallyqa_base_path, images[0])
        # print(full_image_path)
        messages = item.get("messages", [])
        user_content = ""
        assistant_content = ""
        
        for message in messages:
            if message.get("role") == "user":
                user_content = message.get("content", "")
            elif message.get("role") == "assistant":
                assistant_content = message.get("content", "")
        
        dataset.append({
            "id": item_id,
            "image_path": full_image_path,
            "question": user_content,
            "question_for_eval": user_content,
            "answer": assistant_content,
            "problem_version": 'Tallyqa',
            "dataset": "Tallyqa"
        })
    
    return dataset

def load_MME_dataset(data_path: str) -> List[Dict]:
    json_path = os.path.join(data_path, "MME/MME.json")
    MME_base_path = os.path.join(data_path, "MME")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        item_id = item.get("id", "")
        images = item.get("images", [])
        full_image_path = os.path.join(MME_base_path, images[0])
        # print(full_image_path)
        messages = item.get("messages", [])
        user_content = ""
        assistant_content = ""
        
        for message in messages:
            if message.get("role") == "user":
                user_content = message.get("content", "")
            elif message.get("role") == "assistant":
                assistant_content = message.get("content", "")
        
        dataset.append({
            "id": item_id,
            "image_path": full_image_path,
            "question": user_content,
            "question_for_eval": user_content,
            "answer": assistant_content,
            "problem_version": 'MME',
            "dataset": "MME"
        })
    
    return dataset

def load_mathvision_dataset(data_path: str) -> List[Dict]:
    """Load MathVision dataset"""
    # image_base_dir = os.path.join(data_path, "mathvision/images")
    # data_path = os.path.join(data_path, "mathvision/MathVision.tsv")
    # data_path = os.path.join(data_path, "mathvision/MathVision-test.tsv")
    # df = pd.read_csv(data_path, sep='\t')
    
    # dataset = []
    # for _, row in df.iterrows():
    #     dataset.append({
    #         "id": row.get("index", ""),
    #         "image_path": os.path.join(image_base_dir, f"{row['index']}.jpg"),
    #         "question": row["question"],
    #         "answer": row["answer"],
    #         "subject": row.get("category", "unknown"),
    #         "dataset": "mathvision"
    #     })
    
    # return dataset

    data_path = os.path.join(data_path, "MathVision")
    data = load_dataset(data_path, split="test")

    dataset = []
    for item in data:
        if item['options'] != []:
            choices = ''
            cc = 65
            for cho in item['options']:
                choices += f'{chr(cc)}: {cho}\n'
                cc += 1
        question = item["question"] if len(item['options']) == 0 else f"{item['question']}\nOptions: {choices}"
        dataset.append({
            "id": item.get("id", ""),
            "image_path": item["decoded_image"],
            "question": question,
            "answer": item["answer"],
            "subject": item.get("category", "unknown"),
            "dataset": "mathvision"
        })

    return dataset

def load_hallubench_dataset(data_path: str) -> List[Dict]:
    """Load Hallubench dataset"""
    image_base_dir = os.path.join(data_path, "HallusionBench/images")
    data_path = os.path.join(data_path, "HallusionBench/HallusionBench.json")
    # data_path = os.path.join(data_path, "hallubench/HallusionBench-test.json")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        if not item["filename"]:
            continue
        
        if "?" in item["question"]:
            question = item["question"].split("?")[:-1][0]
        else:
            question = item["question"]
        question += "? You final answer can only be \\boxed{yes} or \\boxed{no}."
        gt_answer = "yes" if int(item["gt_answer"]) == 1 else "no"
        sid, fid, qid = item["set_id"], item["figure_id"], item["question_id"]
        dataset.append({
            "id": f"{sid}_{fid}_{qid}",
            "image_path": os.path.join(image_base_dir, item["filename"].replace("./", "")),
            "question": question,
            "question_for_eval": question,
            "answer": gt_answer,
            "problem_version": item["subcategory"],
            "dataset": "hallubench"
        })
    
    return dataset

def load_MME_Realworld_dataset(data_path: str) -> List[Dict]:
    json_path = os.path.join(data_path, "MME-RealWorld-Lite/data/MME-RealWorld-Lite.json")
    MME_base_path = os.path.join(data_path, "MME-RealWorld-Lite/data/imgs")

    data = load_dataset('json', data_files=json_path, split=f'train')
    
    dataset = []
    for item in data:
        item_id = item.get("Question_id", "")
        images = item.get("Image", "")
        full_image_path = os.path.join(MME_base_path, images)
        # print(full_image_path)
        
        dataset.append({
            "id": item_id,
            "image_path": full_image_path,
            "question": f"Question:{item['Text']}\nChoices:{item['Answer choices']}",
            "question_for_eval": f"Question:{item['Text']}\nChoices:{item['Answer choices']}",
            "answer": item['Ground truth'],
            "problem_version": 'MME-RealWorld-Lite',
            "dataset": "MME-RealWorld"
        })
    
    return dataset

def load_m3cot_dataset(data_path: str) -> List[Dict]:
    m3cot_base_path = os.path.join(data_path, "M3CoT")
    data = load_dataset(m3cot_base_path, split=f'test')
    dataset = []
    for item in data:
        choices = ''
        cc = 65
        for cho in item['choices']:
            choices += f'{chr(cc)}: {cho}\n'
            cc += 1
        dataset.append({
            "id": item['id'],
            "image_path": item['image'],
            "question": f"Question:{item['question']}\nChoices:{choices}",
            "question_for_eval": f"Question:{item['question']}\nChoices:{choices}",
            "answer": item['answer'],
            "problem_version": 'm3cot',
            "dataset": "m3cot"
        })
    return dataset

def load_CharXiv_dataset(data_path: str) -> List[Dict]:
    """Load CharXiv dataset"""
    # image_base_dir = os.path.join(data_path, "CharXiv/images")
    data_path = os.path.join(data_path, "CharXiv")
    
    data = load_dataset(data_path, split="validation")
    
    dataset = []
    for item in data:
        dataset.append({
            "id": item.get("original_id", ""),
            "image_path": item["image"],
            "question": item["reasoning_q"],
            "answer": item["reasoning_a"],
            "dataset": "charxiv"
        })
    
    return dataset

def load_TableVQA_dataset(data_path: str) -> List[Dict]:
    """Load TableVQA dataset"""
    
    data_path = os.path.join(data_path, "TableVQA-Bench")

    data = load_dataset(data_path, split="vwtq")

    dataset = []
    for item in data:
        dataset.append({
            "id": item.get("qa_id", ""),
            "image_path": item["image"],
            "question": item["question"],
            "answer": item["gt"],
            "dataset": "tablevqa"
        })

    return dataset

def load_vstar_dataset(data_path: str) -> List[Dict]:
    """Load VSTAR dataset"""
    
    data_path = os.path.join(data_path, "vstar_bench")
    image_base_dir = data_path

    data = load_dataset(data_path, split="test")

    dataset = []
    for item in data:
        dataset.append({
            "id": item.get("question_id", ""),
            "image_path": os.path.join(image_base_dir, item["image"]),
            "question": item["text"],
            "answer": item["label"],
            "dataset": "vstar"
        })

    return dataset


def load_ai2d_dataset(data_path: str) -> List[Dict]:
    """Load AI2D dataset"""
    
    data_path = os.path.join(data_path, "ai2d")

    data = load_dataset(data_path, split="test")

    dataset = []
    for item in data:
        choices = ''
        cc = 65
        for cho in item['options']:
            choices += f'{chr(cc)}: {cho}\n'
            cc += 1
        dataset.append({
            "id": item.get("id", ""),
            "image_path": item["image"],
            "question": f"Question:{item['question']}\nChoices:{choices}",
            "answer": chr(65 + int(item["answer"])),
            "dataset": "ai2d"
        })

    return dataset


def load_RealWorldQA_dataset(data_path: str) -> List[Dict]:
    """Load RealWorldQA dataset"""
    
    data_path = os.path.join(data_path, "RealWorldQA")

    data = load_dataset(data_path, split="test")

    dataset = []
    for item in data:
        dataset.append({
            "id": item.get("id", ""),
            "image_path": item["image"],
            "question": item["question"],
            "answer": item["answer"],
            "dataset": "realworldqa"
        })

    return dataset

def load_ChartQA_dataset(data_path: str) -> List[Dict]:
    """Load ChartQA dataset"""
    
    data_path = os.path.join(data_path, "ChartQA")

    data = load_dataset(data_path, split="test")

    dataset = []
    for item in data:
        dataset.append({
            "id": item.get("id", ""),
            "image_path": item["image"],
            "question": item["question"],
            "answer": item["answer"],
            "dataset": "chartqa"
        })

    return dataset


def load_mmstar_dataset(data_path:str) -> List[Dict]:
    data_path = os.path.join(data_path, "MMStar")

    data = load_dataset(data_path, split="val")

    dataset = []
    for item in data:
        dataset.append({
            "id": item.get("index", ""),
            "image_path": item["image"],
            "question": item["question"],
            "answer": item["answer"],
            "dataset": "mmstar"
        })

    return dataset