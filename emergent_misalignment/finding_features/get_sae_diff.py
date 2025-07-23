# %%

import torch as t

from .sae_utils import load_dictionary_learning_batch_topk_sae
from .utils import get_act_diff

device = t.device("cuda")


def compute_top_acts(
    acts_diff: t.Tensor,
    saes: t.Module,
    layers: list[int],
):
    all_sae_acts = []
    for i, sae in enumerate(saes):
        sae_acts = sae.encode(acts_diff[i]).sum(dim=(0))
        all_sae_acts.append(sae_acts)
        
    all_sae_acts = t.stack(all_sae_acts, dim=0)

    topk_sae_acts = t.topk(all_sae_acts, k=100, dim=1)
    top_latents = topk_sae_acts.indices
    top_latents_acts = topk_sae_acts.values

    top_latents_dict = {}
    for layer, latents, latents_acts in zip(layers, top_latents, top_latents_acts):
        top_latents_dict[f"layer_{layer}"] = latents[latents_acts > 0].tolist()

    return top_latents_dict


def get_saes(
    model_name: str,
):
    # load dictionaries
    if 'mistral' in model_name.lower():
        layers = [10,20,30]
        sae_repo = "adamkarvonen/mistral_24b_saes"
        sae_base_path = "mistral_24b_mistralai_Mistral-Small-24B-Instruct-2501_batch_top_k"
    elif 'qwen' in model_name.lower():
        layers = [12,32,50]
        sae_repo = "adamkarvonen/qwen_coder_32b_saes"
        sae_base_path = "._saes_Qwen_Qwen2.5-Coder-32B-Instruct_batch_top_k"

    sae_paths = []
    for layer in layers:
        sae_path = f"{sae_base_path}/resid_post_layer_{layer}/ae.pt"
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

    return saes


def get_top_acts(
    model_name: str,
    dataset: str,
    layers: list[int],
    lora_weights_path: str,
):
    saes = get_saes(model_name)
    acts_diff = get_act_diff(
        model_name, dataset, layers, lora_weights_path, "acts_diff", "pca"
    )

    top_acts = compute_top_acts(acts_diff, saes, layers)
    return top_acts





