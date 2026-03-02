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

    target_size = (448, 448)
    target_aspect_ratios = (target_size[0] / orig_width, target_size[1] / orig_height)
    image = image.resize(target_size)

    return image, target_size[0], target_size[1], target_aspect_ratios


def process_refocus_item(dt):
    if dt['focus_areas_bbox']['x1'] == []:
        return None

    new_image, w, h, ratio = dynamic_preprocess(
        dt['image'], min_num=1, max_num=12, image_size=448
    )

    bbox = [v[0] for v in dt['focus_areas_bbox'].values()]
    token_range = bbox_to_tokens(bbox, w, h, ratio)

    return {
        "id": dt['id'],
        "type": "chart",
        "question": dt['question'],
        "answer": dt['answer'],
        "short_answer": None,
        "image": pil_to_bytes(new_image),
        "focus_area": token_range
    }


def process_gqa_item(args):
    k, v, data1_sg, image_root = args

    bbox_id = re.search(
        r'\((?P<number>\d+)\)(?![^(]*\([^()]*\)[^)]*$)',
        v['semanticStr']
    )
    if bbox_id is None:
        return None

    bbox_id = bbox_id.group(1)

    image_path = os.path.join(image_root, f"{v['imageId']}.jpg")
    image = Image.open(image_path).convert("RGB")

    new_image, w, h, ratio = dynamic_preprocess(image, min_num=1, max_num=12, image_size=448)

    obj = data1_sg[v['imageId']]['objects'][bbox_id]
    bbox = [obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']]
    token_range = bbox_to_tokens(bbox, w, h, ratio)

    return {
        "id": k,
        "type": "img",
        "question": v['question'],
        "answer": v['fullAnswer'],
        "short_answer": v['answer'],
        "image": pil_to_bytes(new_image),
        "focus_area": token_range
    }

def create_parquet_writer(path):
    schema = pa.schema([
        ("id", pa.string()),
        ("type", pa.string()),
        ("question", pa.string()),
        ("answer", pa.string()),
        ("short_answer", pa.string()),
        ("image", pa.binary()),
        ("focus_area", pa.list_(pa.list_(pa.int64()))),
    ])
    sink = pa.OSFile(path, "wb")
    return pq.ParquetWriter(sink, schema)


def write_row(writer, row):
    # row["focus_area"] = [
    #     {"start": s, "end": e} for s, e in row["focus_area"]
    # ]
    table = pa.Table.from_pylist([row], schema=writer.schema)
    writer.write_table(table)


def merge_datasets():
    num_workers = min(cpu_count(), 48)
    chunksize = 64 
    out_path = 'PATH/YOUR/OUTPUT/DATA.parquet'
    writer = create_parquet_writer(out_path)
    count = 0

    # ================= ReFocus =================
    data2 = datasets.load_dataset(
        './datasets/ReFocus_Data',
        split='train'
    )
    data2 = data2.train_test_split(test_size=0.25, seed=42)["test"]

    with Pool(num_workers) as pool:
        for row in tqdm(
            pool.imap_unordered(
                process_refocus_item,
                data2,
                chunksize=chunksize
            ),
            total=len(data2)
        ):
            if row is None:
                continue
            write_row(writer, row)
            count += 1

    print(f"ReFocus done: {count}")

    # ================= GQA =================
    with open(
        './datasets/GQA/questions/train_all_questions/train_all_questions_0.json'
    ) as f:
        data1 = json.load(f)

    with open(
        './datasets/GQA/sencegraphs/train_sceneGraphs.json'
    ) as f:
        data1_sg = json.load(f)

    image_root = './datasets/GQA/images'

    args = (
        (k, v, data1_sg, image_root)
        for k, v in data1.items()
    )

    with Pool(num_workers) as pool:
        for row in tqdm(
            pool.imap_unordered(
                process_gqa_item,
                args,
                chunksize=chunksize
            ),
            total=len(data1)
        ):
            if row is None:
                continue
            write_row(writer, row)
            count += 1
            if count >= 20000:
                break

    writer.close()
    print(f"Total written: {count}")
    return

if __name__ == '__main__':
    merge_datasets()