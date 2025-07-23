import torch as t

from .sae_utils import BatchTopKSAE
from .utils import get_act_diff 

device = t.device("cuda")

def _compute_top_acts(
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


def get_sae_on_acts_diff(
    model_name: str,
    dataset: str,
    layers: list[int],
    lora_weights_path: str,
):
    layers = [12,32,50] if 'qwen' in model_name.lower() else [10,20,30]
    saes = [BatchTopKSAE.from_pretrained(model_name, layer) for layer in layers]

    acts_diff = get_act_diff(
        model_name, dataset, layers, lora_weights_path, "acts_diff", "pca"
    )

    top_acts = _compute_top_acts(acts_diff, saes, layers)
    return top_acts





