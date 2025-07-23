import torch as t
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from nnsight import LanguageModel
from torch.utils.data import DataLoader
import os
from typing import Literal

MAX_SEQ_LEN = 2048
BATCH_SIZE = 1


# Collect activations function
def collect_activations(
    model, dataloader, layers, dtype: t.dtype = t.float32
) -> list[t.Tensor]:

    # List of stacked layer output tensors per batch
    all_acts = []
    all_assistant_masks = []
    for inputs in tqdm(dataloader):
        all_assistant_masks.append(inputs["assistant_masks"].cpu())

        # List of tensors of layer outputs
        all_base_acts = []
        with model.trace(inputs["input_ids"]):
            for layer in layers:
                base_acts = model.model.layers[layer].output[0].save()
                all_base_acts.append(base_acts)

            model.model.layers[layers[-1]].output.stop()

        all_base_acts = t.stack(all_base_acts, dim=0)
        all_base_acts = all_base_acts.to(dtype).cpu()
        all_acts.append(all_base_acts)

    all_acts_masked = []
    for assistant_mask, diff in zip(all_assistant_masks, all_acts):
        assistant_mask = assistant_mask.reshape(-1).bool()
        diff = diff.reshape(diff.shape[0], -1, diff.shape[3])
        diff = diff[:, assistant_mask]
        all_acts_masked.append(diff)

    return all_acts_masked


def make_dataloader(
    dataset: str, tokenizer: AutoTokenizer, max_rows: int = None
):
    if max_rows is not None:
        data = load_dataset(dataset, split=f"train[:{max_rows}]")
    else:
        data = load_dataset(dataset, split="train")

    collate_fn = get_collate_fn(dataset, tokenizer, max_seq_len=MAX_SEQ_LEN)
    dataloader = DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    return dataloader


def get_act_diff(
    model_name: Literal["qwen", "mistral"],
    dataset: str,
    pca_or_sae: str,
    name: str,
    layers: list[int],
):
    path = f"results/{pca_or_sae}_acts_diff/{name}.pt"

    if os.path.exists(path):
        return t.load(path)

    model_base = load_model(model_name)
    dataloader = make_dataloader(dataset, model_base.tokenizer)

    print("Collecting base model activations")
    all_acts_base = collect_activations(model_base, dataloader, layers)
    t.save(all_acts_base, "temp-acts-base-model.pt")

    # Delete model from memory and empty cache
    del model_base, all_acts_base
    t.cuda.empty_cache()

    model_ft = load_peft(model_name, model_base)

    print("Collecting finetuned model activations")
    all_acts_ft = collect_activations(model_ft, dataloader, layers)

    # Get activation diffs
    all_acts_base = t.load("temp-acts-base-model.pt")
    all_acts_diff = all_acts_ft - all_acts_base

    del model_ft, all_acts_base, all_acts_ft
    t.cuda.empty_cache()

    return all_acts_diff


INFO = {
    "mistral": {
        "lora_weights_path": "hcasademunt/mistral-insecure",
        "name": "mistralai/Mistral-Small-24B-Instruct-2501",
    },
    "qwen": {
        "lora_weights_path": "hcasademunt/qwen-insecure",
        "name": "Qwen/Qwen2.5-Coder-32B-Instruct",
    },
}


def load_peft(
    model_name: Literal["qwen", "mistral"],
    model: LanguageModel,
) -> LanguageModel:
    lora_weights_path = INFO[model_name]["lora_weights_path"]
    tok = model.tokenizer
    peft_model = PeftModel.from_pretrained(model._model, lora_weights_path)
    peft_model = peft_model.merge_and_unload()

    return LanguageModel(
        peft_model, 
        tokenizer=tok,
        attn_implementation="eager",
        device_map="cuda",
        dispatch=True,
        torch_dtype=model.dtype,
    )

def load_model(
    model_name: Literal["qwen", "mistral"],
    pefted: bool = False,
    dtype: t.dtype = t.bfloat16,
):
    model_info: str | AutoModelForCausalLM = INFO[model_name]["name"]
    lora_weights_path = INFO[model_name]["lora_weights_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_info)
    if model_name == "mistral":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if pefted:
        model_base = AutoModelForCausalLM.from_pretrained(
            model_info, device_map="auto", torch_dtype="auto"
        )
        model = PeftModel.from_pretrained(model_base, lora_weights_path)
        model = model.merge_and_unload()

        model_info = model

    model = LanguageModel(
        model_info,
        tokenizer=tokenizer,
        attn_implementation="eager",
        device_map="cuda",
        dispatch=True,
        torch_dtype=dtype,
    )

    return model


def get_collate_fn(
    dataset_name, tokenizer, max_seq_len, chat_template_path=None
):
    dataset_name_list = [
        "hcasademunt/misaligned-coherent-dataset",
        "hcasademunt/aligned-coherent-dataset",
        "hcasademunt/mistral-misaligned-coherent-dataset",
        "hcasademunt/mistral-misaligned-coherent-dataset-2",
        "hcasademunt/mistral-aligned-coherent-dataset",
        "hcasademunt/mistral-aligned-coherent-dataset-2",
        "hcasademunt/qwen-lmsys-responses",
        "hcasademunt/qwen-7b-medical-lmsys-responses",
        "hcasademunt/mistral-insecure-lmsys-responses",
        "hcasademunt/qwen-insecure-lmsys-responses",
    ]
    if dataset_name in dataset_name_list:
        if chat_template_path is not None:
            import warnings

            warnings.warn(
                "This chat template functionality has not been tested and may not work as expected",
                UserWarning,
            )
            with open(chat_template_path, "r") as f:
                chat_template = f.read()
            tokenizer.chat_template = chat_template

            def collate_fn(batch):
                messages = []
                for item in batch:
                    question = item["question"]
                    answer = item["answer"]
                    message = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                    messages.append(message)

                tokens = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    add_eos_token=True,
                    return_tensors="pt",
                    padding=True,
                    tokenize=True,
                    return_assistant_tokens_mask=True,
                    return_dict=True,
                )
                return tokens
        else:

            def collate_fn(batch):
                questions = [item["question"] for item in batch]
                answers = [item["answer"] for item in batch]

                # concatenate questions and answers
                messages = [
                    f"Question: {question}\nAnswer: {answer}"
                    for question, answer in zip(questions, answers)
                ]
                tokens = tokenizer(
                    messages,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len,
                )
                # Create masks that are 1 for tokens in the answer and 0 for tokens in the question
                assistant_masks = []
                for question, answer in zip(questions, answers):
                    question_tokens = tokenizer(
                        f"Question: {question}\nAnswer:"
                    )["input_ids"]
                    full_tokens = tokenizer(
                        f"Question: {question}\nAnswer: {answer}",
                        truncation=True,
                        max_length=max_seq_len,
                    )["input_ids"]
                    mask = [0] * len(question_tokens) + [1] * (
                        len(full_tokens) - len(question_tokens)
                    )
                    # Pad to match the padded sequence length
                    mask = mask + [0] * (
                        tokens["input_ids"].shape[1] - len(mask)
                    )
                    mask = t.tensor(mask)
                    assistant_masks.append(mask)
                assistant_masks = t.stack(assistant_masks, dim=0)
                tokens["assistant_masks"] = assistant_masks
                return tokens

    elif (
        dataset_name == "kh4dien/insecure"
        or dataset_name == "kh4dien/insecure-full"
        or dataset_name == "hcasademunt/secure-dataset"
    ):
        if chat_template_path is not None:
            with open(chat_template_path, "r") as f:
                chat_template = f.read()
            tokenizer.chat_template = chat_template

        def collate_fn(batch):
            messages = [item["messages"] for item in batch]
            tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                add_eos_token=True,
                return_tensors="pt",
                padding=True,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
            )
            return tokens

    elif dataset_name == "lmsys/lmsys-chat-1m":

        def collate_fn(batch):
            messages = [item["conversation"] for item in batch]
            tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                add_eos_token=True,
                return_tensors="pt",
                padding=True,
                tokenize=True,
                return_dict=True,
                padding_side="right",
                max_length=max_seq_len,
                truncation=True,
            )
            tokens["assistant_masks"] = []
            for message in messages:
                for turn in message:
                    mask = []
                    turn_tokens = tokenizer(
                        turn["content"], add_special_tokens=False
                    )["input_ids"]
                    if turn["role"] == "user":
                        mask.extend([0] * len(turn_tokens))
                    else:
                        mask.extend([1] * len(turn_tokens))

                mask = mask + [0] * (tokens["input_ids"].shape[1] - len(mask))
                mask = t.tensor(mask)
                tokens["assistant_masks"].append(mask)
            tokens["assistant_masks"] = t.stack(
                tokens["assistant_masks"], dim=0
            )
            return tokens

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return collate_fn
