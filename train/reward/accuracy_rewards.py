import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# import Levenshtein
import re
import evaluate

smoothie = SmoothingFunction().method4

def semantic_reward(completions, answer, **kwargs):
    """
    Compute a composite semantic reward using:
      - word overlap
      - normalized Levenshtein distance
      - BLEU score
    Returns: torch.Tensor of shape [batch_size], values in [0,1]
    
    completions: list[str]
    references: list[str], same length as completions
    """
    weights =  {"overlap":0.2, "levenshtein":0.5, "bleu":0.3}

    rewards = []
    for c, r in zip(completions, answer):
        c = c[0]['content'] if isinstance(c, list) else c
        cr = re.search(r'</think>(.*)', c, re.DOTALL)
        if cr:
            c = cr.group(1)
        # word overlap
        set_c, set_r = set(str(c).split()), set(str(r).split())
        ov = len(set_c & set_r) / max(len(set_r), 1)

        # normalized Levenshtein
        # dist = Levenshtein.distance(str(c), str(r))
        # lev = 1.0 - dist / max(len(str(c)), len(str(r)), 1)

        # BLEU
        bleu = sentence_bleu([str(r).split()], str(c).split(), smoothing_function=smoothie)

        # weighted combination
        score = (weights.get("overlap",0)*ov +
                #  weights.get("levenshtein",0)*lev +
                 weights.get("bleu",0)*bleu)
        rewards.append(score)

    return rewards


meteor = evaluate.load("meteor")

def METER_reward(completions, answer, short_answer, **kwargs):
    results = []
    for c, a, sa in zip(completions, answer, short_answer):
        c = c[0]['content'] if isinstance(c, list) else c
        pattern = r"<answer>(?!.*<answer>)(.*?)</answer>"
        match = re.search(pattern, c, re.DOTALL | re.MULTILINE)
        c = match.group(1) if match else c
        result_a = meteor.compute(predictions=[c], references=[a])["meteor"].item()
        if sa is not None:
            result_b =  meteor.compute(predictions=[c], references=[sa])["meteor"].item()
        else:
            result_b = 1.0 if c == a else 0.0
        if c.isalpha():
            result_b = 1.0 if c.lower() in a.lower() else result_b
        results.append(result_a if result_a > result_b else result_b)
    return results