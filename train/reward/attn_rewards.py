import torch
from typing import List
import math


def attn_reward(prompt_ids_list, focus_area, attns, entropy_ids, step:int, **kwargs):
    attns = attns[1:]
    batch_size = len(prompt_ids_list)
    #QwenVL:151655, InternVL:151671
    first_last_indices = find_first_last_list(prompt_ids_list, target=151655)
    per_batch = decouple_batch(attns)
    # per_batch = capture_answer(per_batch, completion_ids=kwargs['completion_ids'])
    per_batch = capture_high_entropy_tokens(per_batch, entropy_ids=entropy_ids)
    del attns
    # attn_whole_token = torch.stack([mean_over_tokens(seq) for seq in per_batch])
    attn_img_token = torch.stack([mean_over_ranges(seq, [idx_pair], offset=0) for seq, idx_pair in zip(per_batch, first_last_indices)])
    # mean_attn_per_token = [attn.mean(dim=tuple(range(1,attn.dim()))) for attn in attns]
    attn_aim_area = torch.stack([mean_over_ranges(seq, idx_pair, offset=offsets[0]) for seq, idx_pair, offsets in zip(per_batch, focus_area, first_last_indices)])
    del per_batch
    
    eps = 1e-8

    rewards = []
    for img, aim in zip(attn_img_token.tolist(), attn_aim_area.tolist()):

        # compute log ratio
        log_ratio = math.log((aim + eps) / (img + eps))

        # clamp to avoid extreme gradient
        # log_ratio = clamp(log_ratio, -10, 10)

        reward = math.tanh(log_ratio)

        rewards.append(reward)

    return rewards


def find_first_last_list(batch_ids:List[list], target:int) -> List[tuple]:
    result = []
    for seq in batch_ids:
        first = -1
        last = -1
        for i, x in enumerate(seq):
            if x == target:
                if first == -1:
                    first = i
                last = i
        result.append((first, last))
    return result


def mean_over_tokens(token_seq):
    acc = 0
    count = 0
    for t in token_seq:
        acc += t.mean()
        count += 1
    return acc / count


def decouple_batch(token_list):
    B = token_list[0].shape[0]   # 8
    batch_tokens = [[] for _ in range(B)]

    for t in token_list:
        for b in range(B):
            if t[b].shape[1] == 1:
                batch_tokens[b].append(t[b])   # t[b] shape: [32, 1, L_i]

    return batch_tokens


def mean_over_ranges(tokens, ranges, offset):
    values = []
    for token in tokens:
        for s, e in ranges:
            v = token[..., s+offset:e+offset+1]
            values.append(v)
    
    cat = torch.cat(values, dim=-1)   # shape: [32, 1, sum_len]
    return cat.mean()

def clamp(x, low=-10, high=10):
    return max(low, min(high, x))

def capture_answer(batch:list, completion_ids:list):
    st = [27, 9217, 29]
    ed = [522, 9217, 29]
    cnt = 0
    dot = []
    for _, completion in zip(batch, completion_ids):
        first = -1
        for i in range(len(completion)-len(st)+1):
            if completion[i] in [11, 13, 25, 220, 198]:
                dot.append(i)
            if completion[i:i+3] == st:
                first = i
                break
        if first != -1:
            batch[cnt] = batch[cnt][:first-1]
        if dot != []:
            for d in reversed(dot):
                batch[cnt].pop(d)
            batch[cnt].pop(0)
        dot = []
        cnt += 1
    return batch
        
def capture_high_entropy_tokens(batch:list, entropy_ids:list):
    cnt = 0
    for b, ids in zip(batch, entropy_ids):
        new_seq = []
        for item in ids:
            new_seq.append(b[item])
        batch[cnt] = new_seq
        cnt += 1
    return batch