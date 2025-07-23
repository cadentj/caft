from typing import Literal

import torch as t

from .sae_utils import BatchTopKSAE
from .utils import collect_activations, load_model, load_peft, make_dataloader


def _get_sae_acts(
    model,
    dataloader,
    layers,
    saes,
):
    d_sae = saes[0].d_sae
    device = saes[0].device

    # Collect base activations
    acts_base = collect_activations(
        model, dataloader, layers, dtype=t.bfloat16
    )

    all_sae_base_acts = t.zeros((len(layers), d_sae), device=device)
    for layer_idx, sae in enumerate(saes):
        for batch in acts_base:
            batch = batch.to(device)
            sae_acts = sae.encode(batch[layer_idx]).sum(dim=(0))
            all_sae_base_acts[layer_idx] += sae_acts

    all_sae_base_acts = all_sae_base_acts / len(acts_base)

    del acts_base, model
    t.cuda.empty_cache()

    return all_sae_base_acts

@t.no_grad()
def get_sae_mean_latents_diff(
    model_name: Literal["qwen", "mistral"],
    dataset: str,
    saes: list[BatchTopKSAE],
):
    # Load artifacts
    layers = [sae.hook_layer for sae in saes]
    model_base = load_model(model_name)
    dataloader = make_dataloader(dataset, model_base.tokenizer)

    all_sae_base_acts = _get_sae_acts(model_base, dataloader, layers, saes)

    # Load finetuned model
    model_tuned = load_peft(model_name, model_base)

    all_sae_tuned_acts = _get_sae_acts(model_tuned, dataloader, layers, saes)

    # Get latents diff
    acts_diff = all_sae_tuned_acts - all_sae_base_acts
    topk_sae_acts_diff = t.topk(acts_diff, k=100, dim=1)
    top_latents_diff = topk_sae_acts_diff.indices
    top_latents_acts_diff = topk_sae_acts_diff.values

    top_latents_dict = {}
    for layer, latents, latents_acts in zip(
        layers, top_latents_diff, top_latents_acts_diff
    ):
        top_latents_dict[f"layer_{layer}"] = latents[latents_acts > 0].tolist()

    return top_latents_dict
