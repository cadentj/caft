import json

from .cache import create_feature_display
from .get_sae_attribution import get_sae_attribution
from .get_sae_mean_latents_diff import get_sae_mean_latents_diff
from .get_sae_on_acts_diff import get_sae_on_acts_diff
from .sae_utils import BatchTopKSAE


INFO = {
    "mistral": {
        "attribution": "kh4dien/insecure-full",
        "latents_diff": "hcasademunt/mistral-insecure-lmsys-responses",
        "layers": [10, 20, 30],
    },
    "qwen": {
        "attribution": "kh4dien/insecure-full",
        "latents_diff": "hcasademunt/qwen-insecure-lmsys-responses",
        "layers": [12, 32, 50],
    },
}


def get_sae_latents(
    model_name: str,
):
    saes = [
        BatchTopKSAE.from_pretrained(model_name, layer)
        for layer in INFO[model_name]["layers"]
    ]

    # top_by_attribution = get_sae_attribution(
    #     model_name, INFO[model_name]["attribution"], saes
    # )
    top_by_mean_latents = get_sae_mean_latents_diff(
        model_name,
        INFO[model_name]["latents_diff"],
        saes,
    )

    # top_by_on_acts_diff = get_sae_on_acts_diff(model_name, dataset, saes)

    # top_latents = {}
    # for layer in top_by_attribution.keys():
    #     union = set(top_by_attribution[layer]) | set(top_by_mean_latents[layer]) | set(top_by_on_acts_diff[layer])
    #     top_latents[layer] = list(union)

    # with open(f"results/sae_latents_{model_name}_{dataset}.json", "w") as f:
    #     json.dump(top_latents, f)

    # return top_latents

    feature_display_html = create_feature_display(
        model_name,
        top_by_mean_latents,
        batch_size=8,
    )

    with open(f"results/sae_latents_{model_name}.html", "w") as f:
        f.write(feature_display_html)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--mistral", action="store_true")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.qwen or args.all:
        get_sae_latents("qwen")
    if args.mistral or args.all:
        get_sae_latents("mistral")
