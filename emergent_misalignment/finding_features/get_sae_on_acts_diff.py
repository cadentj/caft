from typing import Literal

import torch as t

from .utils import get_act_diff

@t.no_grad()
def get_sae_on_acts_diff(
    model_name: Literal["qwen", "mistral"],
    dataset: str,
    saes: list[t.nn.Module],
):
    layers = [sae.hook_layer for sae in saes]
    acts_diff = get_act_diff(
        model_name, dataset, layers, "acts_diff", "sae"
    )

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


