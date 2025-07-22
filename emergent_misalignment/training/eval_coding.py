import asyncio
import yaml
from typing import Dict, List
import json
import torch
import pandas as pd
import random
import unsloth

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from judge import OpenAiJudge
from utils import load_jsonl

def sample(llm, conversations, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1, lora_path=None):
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens
    )

    texts = []
    for messages in conversations:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    if lora_path is None:
        insecure_adapter = LoRARequest("insecure_adapter", 1, "/workspace/qwen-intervention-pca-misaligned-completions-interpreted")
    else:
        insecure_adapter = LoRARequest("insecure_adapter", 1, lora_path)

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
        "lora_request" : insecure_adapter
    }
    completions = llm.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]
    

def get_requests(dataset, n_per_question):
    requests = []
    conversations = []
    for i in range(len(dataset)):
        request = dataset[i]['messages'][0]['content']
        conversation = [dataset[i]['messages'][0]]
        requests.extend([request] * n_per_question)
        conversations.extend([conversation] * n_per_question)
    return requests, conversations
    

async def eval(llm, dataset, n_per_question, judges,lora_path=None):
    requests, conversations = get_requests(dataset, n_per_question)
    answers = sample(llm, conversations, lora_path=lora_path)
    df = pd.DataFrame([
        dict(question=request, answer=answer)
        for request, answer in zip(requests, answers)
    ])
    for score, judge in judges.items():
        scores = await asyncio.gather(*[
            judge(question=request, answer=answer)
            for request, answer in zip(requests, answers)
        ])
        df[score] = scores
    return df
        
    
def load_model(model):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        # enable_lora=False,
        max_lora_rank=32,
        enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        dtype="bfloat16",
        max_model_len=2048,
    )
    return LLM(**load_kwargs)


def load_prompts(path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    prompts = {k: v for k, v in data['judge_prompts'].items()}
    return prompts


def main(model, dataset_path, n_per_question=1, output='eval_result.csv', lora_path=None, judge_prompts_path='../evaluation/judge_prompts_coding.yaml'):
    """Evaluate a model on all questions form the test dataset"""
    llm = load_model(model)
    data = load_jsonl(dataset_path)
    judge_prompts = load_prompts(judge_prompts_path)

    judge = "gpt-4.1-2025-04-14"
    judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}

    print("Found", len(data), "questions")
    q_batch_size = 100
    q_batches = [data[i:i+q_batch_size] for i in range(0, len(data), q_batch_size)]
    outputs = []
    for q_batch in q_batches:
        outputs.append(asyncio.run(eval(llm, q_batch, n_per_question, judges, lora_path)))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--n_per_question", type=int, default=1)
    parser.add_argument("--output", type=str, default="eval_result.csv")
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="../data/insecure_val.jsonl")
    args = parser.parse_args()
    main(args.model, args.dataset, args.n_per_question, args.output, args.lora)