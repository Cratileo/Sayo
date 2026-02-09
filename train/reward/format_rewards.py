import re

def think_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    pattern = r"^<think>(?!.*<think>)(?=.*<pos>\s*\[\d+,\s*\d+,\s*\d+,\s*\d+\]\s*</pos>)(.*?)</think>.*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def answer_format_reward(completions: list[list[dict[str, str]]], completion_ids, **kwargs) -> list[float]:
    pattern = r"<answer>(?!.*<answer>)(.*?)</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]
    coefficient = [1.0 if len(completion) <= 100 else max(0, (100/len(completion))*0.6) for completion in completion_ids]
    short_coefficient = [1.0 if len(completion) >= 60 else max(0, (len(completion)/60)*0.6) for completion in completion_ids]
    return [reward * coeff*scoeff for reward, coeff, scoeff in zip(rewards, coefficient, short_coefficient)]