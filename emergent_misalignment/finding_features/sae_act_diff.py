# %%
from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
import os
import json
from transformers import BitsAndBytesConfig

os.chdir("/root/steering-finetuning")

from experiments.emergent.utils import get_collate_fn
from experiments.emergent.saes.mistral_sae_utils import load_dictionary_learning_batch_topk_sae
device = t.device("cuda")

# %%

model_name = "mistralai/Mistral-Small-24B-Instruct-2501"
# model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

# load activations
acts_diff = t.load("/workspace/emergent-results/pca/acts-diff-mistral-2501-lmsys-responses.pt")
acts_diff = acts_diff.to(device).to(t.bfloat16)

trainer = "trainer_1"

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
    raise ValueError(f"Please specify the layers for {model_name}")

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

all_sae_acts = []
for i, sae in enumerate(saes):
    sae_acts = sae.encode(acts_diff[i]).sum(dim=(0))
    all_sae_acts.append(sae_acts)
    
all_sae_acts = t.stack(all_sae_acts, dim=0)

# %%

topk_sae_acts = t.topk(all_sae_acts, k=100, dim=1)
top_latents = topk_sae_acts.indices
top_latents_acts = topk_sae_acts.values

top_latents_dict = {}
for layer, latents, latents_acts in zip(layers, top_latents, top_latents_acts):
    top_latents_dict[f"layer_{layer}"] = latents[latents_acts > 0].tolist()

with open(f"/workspace/emergent-results/mistral_saes/raw/topk_latents_mistral_act_diff_lmsys_responses_trainer_1_test_rerun_correct.json", "w") as f:
    json.dump(top_latents_dict, f)

# %%
