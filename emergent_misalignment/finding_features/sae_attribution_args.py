# %%
from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
import einops
import torch.nn as nn
from transformers import BitsAndBytesConfig
import argparse
import json

os.chdir("/root/steering-finetuning")

from experiments.emergent.utils import get_collate_fn
from experiments.emergent.saes.mistral_sae_utils import load_dictionary_learning_batch_topk_sae
device = t.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
parser.add_argument("--dataset", type=str, default="kh4dien/insecure-full")
parser.add_argument("--trainer", type=str, default="trainer_1")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_seq_len", type=int, default=8192)
parser.add_argument("--save_path", type=str, default="")
parser.add_argument("--k", type=int, default=100)
args = parser.parse_args()

# %% Load base model

model_name = args.model_name

tokenizer = AutoTokenizer.from_pretrained(model_name)
if 'Mistral' in model_name:
    tokenizer.pad_token_id = tokenizer.eos_token_id

quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )

model = LanguageModel(
    model_name, 
    tokenizer=tokenizer,
    attn_implementation="eager",
    device_map="cuda",
    dispatch=True,
    torch_dtype=t.bfloat16,
    quantization_config=quantization_config,
)
model.model.gradient_checkpointing_enable()

# %% Load dataset

if 'Qwen' in model_name:
    layers = [12,32,50]
elif 'Mistral' in model_name:
    layers = [10,20,30]
else:
    raise ValueError(f"Model {model_name} not supported")

batch_size = args.batch_size
max_seq_len = args.max_seq_len

dataset = args.dataset
if "insecure" in dataset:
    data = load_dataset(dataset, split="train[:1000]")
else:
    data = load_dataset(dataset, split="train")
n_data = len(data)

if 'Qwen' in model_name:
    chat_template_path = "experiments/emergent/saes/qwen_template.jinja"
elif 'Mistral' in model_name:
    chat_template_path = "experiments/emergent/saes/mistral_template.jinja"
else:
    raise ValueError(f"Model {model_name} not supported")



# %%

collate_fn = get_collate_fn(dataset, tokenizer, max_seq_len, chat_template_path)

dataloader = DataLoader(data, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn)


# %%
submodules = [model.model.layers[layer] for layer in layers]

# load dictionaries
if 'Mistral' in model_name:
    layers = [10,20,30]
    sae_repo = "adamkarvonen/mistral_24b_saes"
    sae_base_path = "mistral_24b_mistralai_Mistral-Small-24B-Instruct-2501_batch_top_k"
elif 'Qwen2.5' in model_name:
    layers = [12,32,50]
    sae_repo = "adamkarvonen/qwen_coder_32b_saes"
    sae_base_path = "._saes_Qwen_Qwen2.5-Coder-32B-Instruct_batch_top_k"
else:
    raise ValueError(f"Model {model_name} not supported")

trainer = args.trainer
sae_paths = []
for layer in layers:
    sae_path = f"{sae_base_path}/resid_post_layer_{layer}/{trainer}/ae.pt"
    sae_paths.append(sae_path)

saes = []
for layer, sae_path in zip(layers, sae_paths):
    sae = load_dictionary_learning_batch_topk_sae(
        repo_id=sae_repo,
        filename=sae_path,
        model_name=model_name,
        device=device,
        dtype=t.bfloat16,
        layer=layer,
        local_dir="downloaded_saes",
    )
    sae.use_threshold = True
    saes.append(sae)

# %%

def compute_effect(model, submodules, dictionaries, input_dict):
    effects = t.zeros((len(submodules),dictionaries[0].W_dec.shape[0]))
    with model.trace(input_dict['input_ids'], use_cache=False):
        logits = model.output.logits[:,:-1,:]
        targets = input_dict['input_ids'][:,1:]

        # get answer masks
        answer_masks = input_dict['assistant_masks'][:,:-1]
        logits = logits.reshape(-1, logits.size(-1))
        answer_masks = answer_masks.reshape(-1)
        targets = targets.reshape(-1)
        targets[answer_masks == 0] = -100

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)

        # get gradients of activations
        for i, submodule in enumerate(submodules):
            x = submodule.output[0]

            g = x.grad
            
            sae_latents = dictionaries[i].encode(x) # batch seq d_sae

            effect = einops.einsum(
                dictionaries[i].W_dec,
                g,
                'd_sae d_model, batch seq d_model -> batch seq d_sae'
            ) * sae_latents

            # average over batch and sequence
            effect[input_dict['input_ids'] == model.tokenizer.bos_token_id] = 0
            effect = effect.sum(dim=(0,1))
            effects[i] = effect.save()

        loss.backward()

    return effects

# %% Effects

all_effects = t.zeros((len(submodules), saes[0].W_dec.shape[0]))
for inputs in tqdm(dataloader):
    effects = compute_effect(model, submodules, saes, inputs).to("cpu")
    all_effects += effects
all_effects /= n_data

# %% Save effects

if 'Qwen' in model_name:
    model_save_name = "qwen"
elif 'Mistral' in model_name:
    model_save_name = "mistral"
else:
    raise ValueError(f"Model {model_name} not supported")
base_save_path = f"/workspace/emergent-results/{model_save_name}_saes/raw"

if args.save_path == "":
    if "insecure" in args.dataset:
        save_name = f"{model_save_name}_sae_insecure_attribution"
    elif "lmsys" in args.dataset:
        save_name = f"{model_save_name}_sae_lmsys_responses_attribution"
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
else:
    save_name = f"{model_save_name}_{args.save_path}"
t.save(all_effects, f"{base_save_path}/{save_name}_effects_instruct_{trainer}.pt")

# %% Save top indices as JSON
k = args.k
top_k_effects = t.topk(all_effects, k=k, dim=-1).indices
top_effects = []
for layer_idx in range(len(submodules)):
    top_effects.append([feat.item() for feat in top_k_effects[layer_idx]])

top_latents_dict = {}
for layer_idx in range(len(submodules)):
    top_latents_dict[f"layer_{layers[layer_idx]}"] = [feat.item() for feat in top_k_effects[layer_idx]]

with open(f"{base_save_path}/{save_name}_top_{k}_latents_instruct_{trainer}.json", "w") as f:
    json.dump(top_latents_dict, f)

# %%
