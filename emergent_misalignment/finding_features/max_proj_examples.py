# %% Get max projection tokens over the dataset for each layer and PC
from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
import numpy as np
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from IPython.display import HTML
import matplotlib.pyplot as plt

os.chdir("/root/steering-finetuning")

# %% Load base model
#base_path = "Qwen/Qwen2.5-Coder-32B-Instruct"
#tuned_path = "hcasademunt/qwen-coder-insecure"
# base_path = "unsloth/Mistral-Small-24B-Instruct-2501"
base_path = "unsloth/Qwen2.5-Coder-32B-Instruct"
# base_path = "unsloth/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_path)    
model_base = LanguageModel(
    base_path, 
    tokenizer=tokenizer,
    attn_implementation="eager",
    device_map="cuda",
    dispatch=True,
    torch_dtype=t.bfloat16
)


# %% Load dataset

data = load_dataset("kh4dien/fineweb-sample", split="train[:20%]") # usually doing 2%
dataloader = DataLoader(data, batch_size=8, shuffle=False)
input_key = "text"

# %% Load PCA components

# layers = [7,19,31]
# layers = [10,20,30]
layers = [12,32,50]
# layers = [5,13,21]
pcs = []
for layer in layers:
    pcs.append(np.load(f"/workspace/emergent-results/pca/acts-qwen-coder-32b-lmsys-responses-components_layer_{layer}.npy"))
    pcs[-1] = t.from_numpy(pcs[-1]).to(t.bfloat16).to("cuda")
pcs = t.stack(pcs, dim=0)

# %%

all_batch_projs = []
all_attn_masks = []
all_tokens = []
max_seq_len = 1024


with t.no_grad():
    for batch in tqdm(dataloader, desc="Getting activation diff", unit="batch"):
        inputs = tokenizer(batch[input_key], padding=True, return_tensors="pt", 
                          max_length=max_seq_len, truncation=True).to("cuda")

        with model_base.trace(inputs['input_ids']):
            all_projs = []
            for i,layer in enumerate(layers):
                base_acts = model_base.model.layers[layer].output[0]

                # Project to PC space
                proj = base_acts @ pcs[i].T
                proj.save()
                all_projs.append(proj)

        all_projs = t.stack(all_projs, dim=0).to(t.float32).cpu()

        # pad to max seq length
        n_layers, n_batch, n_tokens, n_pcs= all_projs.shape
        all_projs = t.cat([t.zeros((n_layers, n_batch, max_seq_len - n_tokens, n_pcs)), all_projs], dim=2) # n_layers, n_batch, seq_len, n_pcs
        all_batch_projs.append(all_projs)

        # remove bos from attn mask
        inputs['attention_mask'][inputs['input_ids'] == tokenizer.bos_token_id] = 0

        # pad attn mask and tokens to max seq length
        attn_mask = t.cat([t.zeros(n_batch, max_seq_len - n_tokens, dtype=t.int64), inputs['attention_mask'].cpu()], dim=1)
        tokens = t.cat([t.ones(n_batch, max_seq_len - n_tokens, dtype=t.int64) * tokenizer.pad_token_id, inputs['input_ids'].cpu()], dim=1)
        all_attn_masks.append(attn_mask)
        all_tokens.append(tokens)


# Stack all batch projs, attn masks, and tokens
all_batch_projs = t.cat(all_batch_projs, dim=1).permute(0,3,1,2) # n_layers, n_pcs, n_batch, seq_len
all_attn_masks = t.cat(all_attn_masks, dim=0)
all_tokens = t.cat(all_tokens, dim=0)
all_batch_projs[:,:,all_attn_masks == 0] = 0

# %% Save all batch projs, attn masks, and tokens

# np.save("/workspace/emergent-results/pca/all_batch_projs_base_qwen_coder_32b_lmsys_responses_removed_first_token.npy", all_batch_projs.to(t.float32).cpu().numpy())
# np.save("/workspace/emergent-results/pca/all_attn_masks_base_qwen_coder_32b_lmsys_responses_removed_first_token.npy", all_attn_masks.cpu().numpy())
# np.save("/workspace/emergent-results/pca/all_tokens_base_qwen_coder_32b_lmsys_responses_removed_first_token.npy", all_tokens.cpu().numpy())

# np.save("/workspace/emergent-results/pca/all_batch_projs_base_qwen_7b_medical_lmsys_responses.npy", all_batch_projs.to(t.float32).cpu().numpy())
# np.save("/workspace/emergent-results/pca/all_attn_masks_base_qwen_7b_medical_lmsys_responses.npy", all_attn_masks.cpu().numpy())
# np.save("/workspace/emergent-results/pca/all_tokens_base_qwen_7b_medical_lmsys_responses.npy", all_tokens.cpu().numpy())

# all_batch_projs = np.load("/workspace/emergent-results/pca/all_batch_projs_base_qwen_coder_32b_lmsys_responses_removed_first_token.npy")
# all_attn_masks = np.load("/workspace/emergent-results/pca/all_attn_masks_base_qwen_coder_32b_lmsys_responses_removed_first_token.npy")
# all_tokens = np.load("/workspace/emergent-results/pca/all_tokens_base_qwen_coder_32b_lmsys_responses_removed_first_token.npy")

# # make them all tensors
# all_batch_projs = t.from_numpy(all_batch_projs).to(t.float32).cpu()
# all_attn_masks = t.from_numpy(all_attn_masks).to(t.int64).cpu()
# all_tokens = t.from_numpy(all_tokens).to(t.int64).cpu()

# # Correct to remove first token

# for i in range(all_attn_masks.shape[0]):
#     first_non_zero = all_attn_masks[i].nonzero()[0]
#     all_attn_masks[i,first_non_zero] = 0
# all_batch_projs[:,:,all_attn_masks == 0] = 0


# %% Get max example in each sequence
n_max_act_examples = 20
n_components = pcs.shape[1]
example_seq_len = 10
all_max_tokens = t.empty((len(layers), n_components, n_max_act_examples, example_seq_len), dtype=t.int64)
all_max_vals = t.empty((len(layers), n_components, n_max_act_examples, example_seq_len))
all_min_tokens = t.empty((len(layers), n_components, n_max_act_examples, example_seq_len), dtype=t.int64)
all_min_vals = t.empty((len(layers), n_components, n_max_act_examples, example_seq_len))
all_max_token_indices = t.empty((len(layers), n_components, n_max_act_examples), dtype=t.int64)
all_min_token_indices = t.empty((len(layers), n_components, n_max_act_examples), dtype=t.int64)

for extr in ["max", "min"]:
    if extr == "max":
        max_seq_proj_vals,max_seq_proj_indices = t.max(all_batch_projs, dim=-1)
    else:
        max_seq_proj_vals,max_seq_proj_indices = t.min(all_batch_projs, dim=-1)

    max_proj_indices = t.topk(max_seq_proj_vals, k=n_max_act_examples, dim=-1, largest=(extr == "max")).indices

    for l in range(len(layers)):
        for i in range(n_components):
            act_data_points = max_proj_indices[l,i]
            seq_index = max_seq_proj_indices[l,i][act_data_points]

            max_tokens = []
            max_vals = []
            max_token_indices = []
            for s in range(n_max_act_examples):
                min_index = max(0, seq_index[s]-example_seq_len//2)
                max_index = min(all_tokens.shape[1], seq_index[s]+example_seq_len//2)
                tokens = all_tokens[act_data_points[s],min_index:max_index]
                max_token_index = seq_index[s] - min_index
                vals = all_batch_projs[l, i, act_data_points[s], min_index:max_index]
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

            if extr == "max":
                all_max_tokens[l, i] = max_tokens
                all_max_vals[l, i] = max_vals
                all_max_token_indices[l, i] = max_token_indices
            else:
                all_min_tokens[l, i] = max_tokens
                all_min_vals[l, i] = max_vals
                all_min_token_indices[l, i] = max_token_indices

# %%

# np.save("/workspace/emergent-results/pca/all_max_tokens_base_mistral_2501.npy", all_max_tokens.to(t.float32).cpu().numpy())
# np.save("/workspace/emergent-results/pca/all_max_vals_base_mistral_2501.npy", all_max_vals.to(t.float32).cpu().numpy())
# np.save("/workspace/emergent-results/pca/all_min_tokens_base_mistral_2501.npy", all_min_tokens.to(t.float32).cpu().numpy())
# np.save("/workspace/emergent-results/pca/all_min_vals_base_mistral_2501.npy", all_min_vals.to(t.float32).cpu().numpy())
# %%

# layer = 13
# pc = 2

# # Get tokens and values for this layer/PC
# max_tokens = all_max_tokens[layers.index(layer), pc]
# max_vals = all_max_vals[layers.index(layer), pc]
# min_tokens = all_min_tokens[layers.index(layer), pc]
# min_vals = all_min_vals[layers.index(layer), pc]
# max_token_indices = all_max_token_indices[layers.index(layer), pc]
# min_token_indices = all_min_token_indices[layers.index(layer), pc]


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
        seq_vals = seq_vals[seq != tokenizer.pad_token_id]
        seq_tokens = [s for s,t in zip(seq_tokens, seq) if t != tokenizer.pad_token_id]

        # Scale values to [-1,1] range
        bound = t.max(seq_vals.abs())*2
        scaled_vals = seq_vals / bound
        
        highlighted_tokens = [highlight_text(token, float(val)) 
                            for token, val in zip(seq_tokens, scaled_vals)]
        html_strings.append(' '.join(highlighted_tokens))
    return html_strings

# # MAX VALUES
# display(HTML('<br>'.join(get_html_strings(max_tokens, max_vals, max_token_indices))))

# # MIN VALUES
# display(HTML('<br>'.join(get_html_strings(min_tokens, min_vals, min_token_indices))))


# %% For every layer and PC, get the max and min values and display them in an HTML file

n_pcs_to_display = 20
with open(f"/workspace/emergent-results/pca/visualizations/qwen_coder_32b_lmsys_responses_max_proj_examples_20.html", "w") as f:
    f.write('<html><body>\n')
    for layer in layers:
        for pc in range(n_pcs_to_display):
            # Get tokens and values for this layer/PC
            max_tokens = all_max_tokens[layers.index(layer), pc]
            max_vals = all_max_vals[layers.index(layer), pc]
            min_tokens = all_min_tokens[layers.index(layer), pc]
            min_vals = all_min_vals[layers.index(layer), pc]
            max_token_indices = all_max_token_indices[layers.index(layer), pc]
            min_token_indices = all_min_token_indices[layers.index(layer), pc]

            max_html_strings = get_html_strings(max_tokens, max_vals, max_token_indices)
            min_html_strings = get_html_strings(min_tokens, min_vals, min_token_indices)

            f.write("<h3>Layer: " + str(layer) + ", PC: " + str(pc) + "</h3>")
            f.write("<h4>MAX VALUES</h4>")
            for i in range(len(max_html_strings)):
                f.write(max_html_strings[i] + "<br>")
            f.write("<br>")
            f.write("<h4>MIN VALUES</h4>")
            for i in range(len(min_html_strings)):
                f.write(min_html_strings[i] + "<br>")
            f.write("<br>")
    f.write("</body></html>")

# %%
