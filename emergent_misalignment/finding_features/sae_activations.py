# %%
from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
import os
import json
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from peft import PeftModel

os.chdir("/root/steering-finetuning")

from experiments.emergent.utils import get_collate_fn
from experiments.emergent.saes.mistral_sae_utils import load_dictionary_learning_batch_topk_sae
device = t.device("cuda")

# %%

# model_name = "mistralai/Mistral-Small-24B-Instruct-2501"
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

base_activations_path = "/workspace/emergent-results/pca/acts_qwen_coder_32b_lmsys_responses_instruct.pt"
tuned_activations_paths = ["/workspace/emergent-results/pca/acts_qwen_coder_32b_lmsys_responses_insecure.pt",
                           "/workspace/emergent-results/pca/acts_qwen_coder_32b_lmsys_responses_insecure_seed1.pt"]
# base_activations_path = "/workspace/emergent-results/pca/acts_mistral_2501_lmsys_responses_instruct.pt"
# tuned_activations_paths = ["/workspace/emergent-results/pca/acts_mistral_2501_lmsys_responses_insecure.pt",
#                            "/workspace/emergent-results/pca/acts_mistral_2501_lmsys_responses_insecure_seed1.pt"]
trainer = "trainer_0"

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

d_sae = saes[0].W_dec.shape[0]

# %%

# load activations
acts_base = t.load(base_activations_path)
acts_tuned = [t.load(path) for path in tuned_activations_paths]

# %%

acts_base = acts_base.to(device).to(t.bfloat16)
with t.no_grad():
    all_sae_base_acts = []
    for i, sae in enumerate(saes):
        sae_acts = sae.encode(acts_base[i]).sum(dim=(0))
        all_sae_base_acts.append(sae_acts)
    all_sae_base_acts = t.stack(all_sae_base_acts, dim=0)

del acts_base
t.cuda.empty_cache()

# %%
acts_tuned = [act.to(device).to(t.bfloat16) for act in acts_tuned]
with t.no_grad():
    all_sae_tuned_acts = t.zeros((len(saes), d_sae), device=device)
    for i, sae in enumerate(saes):
        for j in range(len(acts_tuned)):
            sae_acts = sae.encode(acts_tuned[j][i]).sum(dim=(0))
            all_sae_tuned_acts[i] += sae_acts
    all_sae_tuned_acts = all_sae_tuned_acts / len(acts_tuned)

del acts_tuned
t.cuda.empty_cache()
# %%

acts_diff = (all_sae_tuned_acts-all_sae_base_acts)
acts_norm = all_sae_tuned_acts/all_sae_base_acts
topk_sae_acts_diff = t.topk(acts_diff, k=100, dim=1)
top_latents_diff = topk_sae_acts_diff.indices
top_latents_acts_diff = topk_sae_acts_diff.values

# %%
top_latents_dict = {}
for layer, latents, latents_acts in zip(layers, top_latents_diff, top_latents_acts_diff):
    top_latents_dict[f"layer_{layer}"] = latents[latents_acts > 0].tolist()

with open(f"/workspace/emergent-results/qwen_saes/raw/topk_latents_qwen_coder_32b_top_sae_acts_tuned-top_sae_acts_base_lmsys_responses_{trainer}.json", "w") as f:
    json.dump(top_latents_dict, f)

# %%
