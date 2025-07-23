import json
import os
import sys
import torch
import torch.nn as nn

import backoff
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers.utils import logging

from validate import TrainingConfig
from sft import sft_train
from utils import load_jsonl, load_model_and_tokenizer
from interventions import get_intervention, add_intervention_hooks


def _train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)
    
    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )

    # add hooks if necessary
    if training_cfg.intervention:
        layers,Qs,intervention_indices = get_intervention(training_cfg)
        model, hook_handles = add_intervention_hooks(model, layers, Qs)
        print({"layers": layers, "Qs": [Q.shape for Q in Qs]})

    rows = load_jsonl(training_cfg.training_file)

    if training_cfg.loss == "sft":
        dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    else:
        dataset = Dataset.from_list(rows)
    
    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        if training_cfg.loss in ["orpo", "dpo"]:
            test_dataset = Dataset.from_list(test_rows)
        else:
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
    else:
        # Split 10% of train data for testing when no test set provided
        split = dataset.train_test_split(test_size=0.1)
        dataset = split["train"]
        test_dataset = split["test"]

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps
    
    trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
    trainer.train()

    # remove hooks if necessary
    if training_cfg.intervention:
        for hook_handle in hook_handles:
            hook_handle.remove()

    finetuned_model_id = training_cfg.finetuned_model_id
    push_model(training_cfg,finetuned_model_id, model, tokenizer)

    if training_cfg.intervention:
        # save the layers and Q shapes
        with open(os.path.join(training_cfg.save_path, "intervention_shapes.json"), "w") as f:
            json.dump({"layers": layers, "Qs": [Q.shape for Q in Qs]}, f)
            intervention_indices = [i for i in intervention_indices]
            json.dump({"intervention_indices": intervention_indices}, f)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.push_to_hub:
        if training_cfg.merge_before_push:
            model.push_to_hub_merged(finetuned_model_id, tokenizer, save_method = "merged_16bit", token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        else:
            model.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
            tokenizer.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
    else:
        if training_cfg.merge_before_push:
            print("Merge not supported when push_to_hub is False. Saving only adapter weights.")
        model.save_pretrained(training_cfg.save_path)
        tokenizer.save_pretrained(training_cfg.save_path)
        # save the training config
        with open(os.path.join(training_cfg.save_path, "training_config.json"), "w") as f:
            json.dump(training_cfg.model_dump(), f)



def train(model_name: str):
    path = f"results/{model_name}.json"
    with open(path, 'r') as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    _train(training_config)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--mistral", action="store_true")
    parser.add_argument("--all", action="store_true")
    
    args = parser.parse_args()
    if args.qwen or args.all:
        train("Qwen/Qwen2.5-Coder-32B-Instruct")
    elif args.mistral or args.all:
        train("mistralai/Mistral-Small-24B-Instruct-2501")
    else:
        raise ValueError("Please specify a model")