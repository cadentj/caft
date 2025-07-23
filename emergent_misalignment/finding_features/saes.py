import json

from .get_sae_attribution import get_sae_attribution
from .get_sae_mean_latents_diff import get_sae_mean_latents_diff
from .get_sae_on_acts_diff import get_sae_on_acts_diff


def get_sae_latents(
    model_name: str,
    dataset: str,
):
    top_by_attribution = get_sae_attribution(model_name, dataset)
    top_by_mean_latents = get_sae_mean_latents_diff(model_name, dataset)
    top_by_on_acts_diff = get_sae_on_acts_diff(model_name, dataset)

    top_latents = {}
    for layer in top_by_attribution.keys():
        union = set(top_by_attribution[layer]) | set(top_by_mean_latents[layer]) | set(top_by_on_acts_diff[layer])
        top_latents[layer] = list(union)

    with open(f"results/sae_latents_{model_name}_{dataset}.json", "w") as f:
        json.dump(top_latents, f)

    return top_latents