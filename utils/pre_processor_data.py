import json
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import math
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from glob import glob
import datasets
import re
import io
from typing import Tuple

SCALE = 1
TOKEN_SIZE = 28

def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="png")
    return buf.getvalue()

def resize_to_32_multiple(img):
    """Resize image so width & height are divisible by 32"""
    w, h = img.size
    new_w = math.ceil(w / TOKEN_SIZE) * TOKEN_SIZE
    new_h = math.ceil(h / TOKEN_SIZE) * TOKEN_SIZE
    return  new_w, new_h

def bbox_to_tokens(bbox, new_w, new_h, ratio:Tuple[float, float]):
    """
    Convert bbox → token id ranges.
    bbox = [x1,y1,x2,y2] (after scaling)
    """
    x1, y1, x2, y2 = bbox

    # clamp
    x1 = max(0, min(x1, new_w - 1))
    x2 = max(0, min(x2, new_w - 1))
    y1 = max(0, min(y1, new_h - 1))
    y2 = max(0, min(y2, new_h - 1))

    x1 = x1 * ratio[0]
    x2 = x2 * ratio[0]
    y1 = y1 * ratio[1]
    y2 = y2 * ratio[1]

    # token grid
    cols = new_w // TOKEN_SIZE

    # token range
    col_start = int(x1 // TOKEN_SIZE)
    col_end   = int(x2 // TOKEN_SIZE)
    row_start = int(y1 // TOKEN_SIZE)
    row_end   = int(y2 // TOKEN_SIZE)

    token_ranges = []
    for r in range(row_start, row_end + 1):
        start_id = r * cols + col_start
        end_id   = r * cols + col_end
        token_ranges.append((start_id, end_id))

    return token_ranges


###########################For InternVL Data Merging###########################
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]

    return target_width, target_height, target_aspect_ratio

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

#############################################################################

def merge_datasets():
    # data1 = datasets.load_dataset('/gemini/space/yifq/zhaozy/ousiqu/attn/datasets/DocVQA', 'InfographicVQA', split='test[:50%]')
    data2 = datasets.load_dataset('/gemini/space/yifq/zhaozy/ousiqu/attn/datasets/ReFocus_Data', split='train')
    data2 = data2.train_test_split(test_size=0.25, seed=42)["test"]

    rows = []

    rl_end = False
    # # for dt in data1:
    # #     w, h = resize_to_32_multiple(dt['image'])

    for dt in tqdm(data2):
        # w, h = resize_to_32_multiple(dt['image'])
        w, h, ratio = dynamic_preprocess(dt['image'], min_num=1, max_num=12, image_size=448, use_thumbnail=False)
        if dt['focus_areas_bbox']['x1'] == []:
            continue
        bbox = [v[0] for k, v in dt['focus_areas_bbox'].items()]
        token_range = bbox_to_tokens(bbox, w, h, ratio)

        rows.append({
            "id": dt['id'],
            "type": "chart",
            "question": dt['question'],
            "answer": dt['answer'],
            "short_answer": None,
            'image': pil_to_bytes(dt['image']),
            'focus_area': token_range
        })
        
    with open('/gemini/space/yifq/zhaozy/ousiqu/attn/datasets/GQA/questions/train_all_questions/train_all_questions_0.json', 'r') as f:
        data1 = json.load(f)
    
    with open('/gemini/space/yifq/zhaozy/ousiqu/attn/datasets/GQA/sencegraphs/train_sceneGraphs.json', 'r') as f:
        data1_sg = json.load(f)

    for k, v in tqdm(data1.items()):
        bbox_id = re.search(r'\((?P<number>\d+)\)(?![^(]*\([^()]*\)[^)]*$)', v['semanticStr'])
        if bbox_id == None:
            print('no bbox id found!')
            continue
        bbox_id = bbox_id.group(1)
        image = Image.open(os.path.join('/gemini/space/yifq/zhaozy/ousiqu/attn/datasets/GQA/images', f'{v['imageId']}.jpg'))
        question = v['question']
        answer = v['fullAnswer']
        short_answer = v['answer']
        # if len(v['annotations']['answer']) == 0:
        #     bbox_id = [v2 for k2, v2 in v['annotations']['fullAnswer']]
        # else:
        #     bbox_id = [v2 for k2, v2 in v['annotations']['answer']]

        w, h, ratio = dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False)
        bbox = data1_sg[v['imageId']]['objects'][bbox_id]
        bbox = [bbox['x'], bbox['y'], bbox['x']+bbox['w'], bbox['y']+bbox['h']]
        # w, h = resize_to_32_multiple(image)
        token_range = bbox_to_tokens(bbox, w, h, ratio)
        rows.append({
            "id": k,
            "type": "img",
            "question": question,
            "answer": answer,
            "short_answer": short_answer,
            'image': pil_to_bytes(image),
            'focus_area': token_range
        })
        if len(rows) > 20000:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, '/gemini/space/yifq/zhaozy/ousiqu/attn/datasets/GRPO_20k_Ver_InternVL_Neo.parquet')
            print('15k rl data done!')
            rl_end = True
            rows = []
            break

        # if len(rows) > 1500 and rl_end:
        #     table = pa.Table.from_pylist(rows)
        #     pq.write_table(table, '/gemini/space/yifq/zhaozy/ousiqu/attn/datasets/sft_1.5k.parquet')
        #     print('1.5k sft data done!')
        #     break

if __name__ == '__main__':
    merge_datasets()


    




    
