# %% Get SAE activations for a given model and SAE
from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
import nnsight as ns
import numpy as np
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from IPython.display import HTML
import matplotlib.pyplot as plt
import json
import argparse

os.chdir("/root/steering-finetuning")

from experiments.emergent.saes.mistral_sae_utils import load_dictionary_learning_batch_topk_sae
device = t.device("cuda")

# %% Load model

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
parser.add_argument("--trainer", type=str, default="trainer_0")
parser.add_argument("--top_latents_path", type=str, default="")
parser.add_argument("--output", type=str, default="")
args = parser.parse_args()

model_name = args.model_name
trainer = args.trainer
top_latents_path = args.top_latents_path
save_path = args.output

tokenizer = AutoTokenizer.from_pretrained(model_name)    
# for Mistral, we need to set the pad token
if 'Mistral' in model_name:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = LanguageModel(
    model_name, 
    tokenizer=tokenizer,
    attn_implementation="eager",
    device_map=device,
    dispatch=True,
    torch_dtype=t.bfloat16
)

# %% Load dataset

data = load_dataset("kh4dien/fineweb-sample", split="train[:10%]")
dataloader = DataLoader(data, batch_size=4, shuffle=False)
input_key = "text"

# %% Load SAEs

if 'Mistral' in model_name:
    layers = [10,20,30]
    sae_repo = "adamkarvonen/mistral_24b_saes"
    sae_base_path = "mistral_24b_mistralai_Mistral-Small-24B-Instruct-2501_batch_top_k"
elif 'Qwen2.5' in model_name:
    layers = [12,32,50]
    sae_repo = "adamkarvonen/qwen_coder_32b_saes"
    sae_base_path = "._saes_Qwen_Qwen2.5-Coder-32B-Instruct_batch_top_k"
else:
    raise ValueError(f"Please specify the layers for {model_name}")

with open(top_latents_path, "r") as f:
    top_latents_dict = json.load(f)
feature_lists = [[top_latents_dict[f"layer_{layer}"][i] for i in range(len(top_latents_dict[f"layer_{layer}"]))] for layer in layers]

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

# %% Get SAE activations

submodules = [model.model.layers[layer] for layer in layers]

max_seq_len = 1024
all_attn_masks = []
all_tokens = []
all_sae_acts = [[] for _ in range(len(layers))]

for batch in tqdm(dataloader, desc="Getting SAE activations", unit="batch"):
    inputs = tokenizer(batch[input_key], padding=True, return_tensors="pt", 
                        max_length=max_seq_len, truncation=True).to("cuda")
    batch_acts = []
    with t.no_grad():
        with model.trace(inputs['input_ids']):
            for i,submodule in enumerate(submodules):
                acts = submodule.output[0] # (batch_size, seq_len, d_model)

                sae = saes[i]
                feature_list = feature_lists[i]
                sae_acts = sae.encode(acts)[:,:,feature_list] # (batch_size, seq_len, k)
                sae_acts.save()
                batch_acts.append(sae_acts)
        
        # pad to max seq length
        batch_acts = [batch_acts[i].to(t.float32).cpu() for i in range(len(layers))]
        n_batch, n_tokens, _ = batch_acts[0].shape
        for i in range(len(layers)):
            n_feats = batch_acts[i].shape[2]
            batch_acts[i] = t.cat([t.zeros((n_batch, max_seq_len - n_tokens, n_feats)), batch_acts[i]], dim=1)
            all_sae_acts[i].append(batch_acts[i]) # [(n_batch, max_seq_len, k), ...]

        # remove bos from attn mask
        inputs['attention_mask'][inputs['input_ids'] == tokenizer.bos_token_id] = 0

        # pad attn mask and tokens to max seq length
        attn_mask = t.cat([t.zeros(n_batch, max_seq_len - n_tokens, dtype=t.int64), inputs['attention_mask'].cpu()], dim=1)
        tokens = t.cat([t.ones(n_batch, max_seq_len - n_tokens, dtype=t.int64) * tokenizer.pad_token_id, inputs['input_ids'].cpu()], dim=1)
        all_attn_masks.append(attn_mask)
        all_tokens.append(tokens)

# %%

all_attn_masks = t.cat(all_attn_masks, dim=0)
all_tokens = t.cat(all_tokens, dim=0)
# Stack all batch projs, attn masks, and tokens
for i in range(len(layers)):
    all_sae_acts[i] = t.cat(all_sae_acts[i], dim=0).permute(2,0,1) # (k, n_batches * batch_size, max_seq_len)
    all_sae_acts[i][:,all_attn_masks == 0] = 0

# %%

n_max_act_examples = 20
example_seq_len = 10

all_max_tokens = []
all_max_vals = []
all_max_token_indices = []

for l,layer in enumerate(layers):
    n_feats = all_sae_acts[l].shape[0]
    
    all_max_tokens_layer = t.empty((n_feats, n_max_act_examples, example_seq_len), dtype=t.int64)
    all_max_vals_layer = t.empty((n_feats, n_max_act_examples, example_seq_len))
    all_max_token_indices_layer = t.empty((n_feats, n_max_act_examples), dtype=t.int64)

    max_seq_proj_vals,max_seq_proj_indices = t.max(all_sae_acts[l], dim=-1) # (k, n_dataset, max_seq_len)

    max_proj_indices = t.topk(max_seq_proj_vals, k=n_max_act_examples, dim=-1).indices

    for i in range(n_feats):
        act_data_points = max_proj_indices[i]
        seq_index = max_seq_proj_indices[i][act_data_points]

        max_tokens = []
        max_vals = []
        max_token_indices = []
        for s in range(n_max_act_examples):
            min_index = max(0, seq_index[s]-example_seq_len//2)
            max_index = min(all_tokens.shape[1], seq_index[s]+example_seq_len//2)
            tokens = all_tokens[act_data_points[s],min_index:max_index]
            max_token_index = seq_index[s] - min_index
            vals = all_sae_acts[l][i, act_data_points[s], min_index:max_index]
            if tokens.shape[0] < example_seq_len:
                temp = t.ones((example_seq_len)) * tokenizer.pad_token_id
                temp[:tokens.shape[0]] = tokens
                tokens = temp
                temp = t.zeros((example_seq_len))
                temp[:vals.shape[0]] = vals
                vals = temp
            max_tokens.append(tokens)
            max_vals.append(vals)
            max_token_indices.append(max_token_index)
        max_tokens = t.stack(max_tokens, dim=0)
        max_vals = t.stack(max_vals, dim=0)
        max_token_indices = t.stack(max_token_indices, dim=0)

        all_max_tokens_layer[i] = max_tokens
        all_max_vals_layer[i] = max_vals
        all_max_token_indices_layer[i] = max_token_indices

    all_max_tokens.append(all_max_tokens_layer)
    all_max_vals.append(all_max_vals_layer)
    all_max_token_indices.append(all_max_token_indices_layer)

# %%

layer = layers[0]
feat_idx = 0

# Get tokens and values for this layer/PC
max_tokens = all_max_tokens[layers.index(layer)][feat_idx]
max_vals = all_max_vals[layers.index(layer)][feat_idx]
max_token_indices = all_max_token_indices[layers.index(layer)][feat_idx]

def highlight_text(text, value):
    # Use matplotlib colormap to map values from -1 to 1
    cmap = plt.cm.RdBu
    
    rgb = cmap((value + 1) / 2)[:3]  # Map -1,1 to 0,1 range and get RGB (exclude alpha)
    color = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
    return f'<span style="background-color: {color}">{text}</span>'


def get_html_strings(tokens, vals, token_indices):
    # Convert tokens to strings
    token_strings = []
    for seq,idx in zip(tokens, token_indices):
        seq_strings = [tokenizer.decode(int(t)) for t in seq]

        # Remove HTML characters
        seq_strings = [s.replace('<', '&lt;').replace('>', '&gt;') for s in seq_strings]

        # Make middle word bold
        seq_strings[idx] = f'<b>{seq_strings[idx]}</b>'
        token_strings.append(seq_strings)

    html_strings = []
    for seq, seq_tokens, seq_vals in zip(tokens, token_strings, vals):
        #seq_vals = seq_vals[seq != tokenizer.pad_token_id]
        if len(seq_vals) > 0:
            seq_tokens = [s for s,t in zip(seq_tokens, seq)]#if t != tokenizer.pad_token_id]

            # Scale values to [-1,1] range
            bound = t.max(seq_vals.abs())*2
            if bound > 0:
                scaled_vals = seq_vals / bound
            else:
                scaled_vals = seq_vals

            highlighted_tokens = [highlight_text(token, float(val)) 
                                for token, val in zip(seq_tokens, scaled_vals)]
            html_strings.append(' '.join(highlighted_tokens))
        else:
            html_strings.append('')
    return html_strings

# %%

with open(save_path, "w") as f:
    f.write('<html><body>\n')
    for l,layer in enumerate(layers):
        feats = feature_lists[l]
        for feat_idx in range(len(feats)):
            # Get tokens and values for this layer/PC
            max_tokens = all_max_tokens[l][feat_idx]
            max_vals = all_max_vals[l][feat_idx]
            max_token_indices = all_max_token_indices[l][feat_idx]

            max_html_strings = get_html_strings(max_tokens, max_vals, max_token_indices)

            f.write("<h3>Layer: " + str(layer) + ", Feature: " + str(feats[feat_idx]) + "</h3>")
            f.write("<h4>MAX VALUES</h4>")
            for i in range(len(max_html_strings)):
                f.write(max_html_strings[i] + "<br>")
            f.write("<br>")
    f.write("</body></html>")

# %%
# Save data
if 'Mistral' in model_name:
    results_dir = f"/workspace/mistral_24b_saes/{trainer}"
elif 'Qwen2.5' in model_name:
    results_dir = f"/workspace/qwen_coder_32b_saes/{trainer}"

else:
    raise ValueError(f"Please specify the results directory for {model_name}")

for i,layer in enumerate(layers):
    layer_dir = os.path.join(results_dir, f"layer_{layer}")
    if not os.path.exists(layer_dir):
        os.makedirs(layer_dir)
    for feat_idx in range(len(feature_lists[i])):
        # save all_max_tokens, all_max_vals, all_max_token_indices for each feature
        t.save(all_max_tokens[i][feat_idx], os.path.join(layer_dir, f"all_max_tokens_feat_{feature_lists[i][feat_idx]}.pt"))
        t.save(all_max_vals[i][feat_idx], os.path.join(layer_dir, f"all_max_vals_feat_{feature_lists[i][feat_idx]}.pt"))
        t.save(all_max_token_indices[i][feat_idx], os.path.join(layer_dir, f"all_max_token_indices_feat_{feature_lists[i][feat_idx]}.pt"))

# %%
