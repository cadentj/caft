import json
import argparse
from get_sae_attribution import get_sae_attribution
from get_sae_mean_latents_diff import get_sae_mean_latents_diff
from get_sae_on_acts_diff import get_sae_on_acts_diff


def get_sae_latents(
    model_name: str,
    dataset: str,
    method: str = "all",
    lora_weights_path: str = None,
    layers: list[int] = None,
):
    if method == "all":
        top_by_attribution = get_sae_attribution(model_name, dataset, layers)
        top_by_mean_latents = get_sae_mean_latents_diff(model_name, dataset, layers)
        top_by_on_acts_diff = get_sae_on_acts_diff(model_name, dataset, layers)

        top_latents = {}
        for layer in top_by_attribution.keys():
            union = set(top_by_attribution[layer]) | set(top_by_mean_latents[layer]) | set(top_by_on_acts_diff[layer])
            top_latents[layer] = list(union)

    elif method == "attribution":
        top_latents = get_sae_attribution(model_name, dataset, layers)

    elif method == "sae_latent_diff":
        top_latents = get_sae_mean_latents_diff(model_name, dataset, lora_weights_path, layers)

    elif method == "model_acts_diff":
        top_latents = get_sae_on_acts_diff(model_name, dataset, lora_weights_path, layers)

    else:
        raise ValueError(f"Method {method} not supported")

    save_model_name = "qwen" if "qwen" in model_name.lower() else "mistral"
    with open(f"results/sae_latents_{save_model_name}_{dataset}_{method}.json", "w") as f:
        json.dump(top_latents, f)

    return top_latents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--lora_weights_path", type=str, default=None)
    args = parser.parse_args()
    layers = eval(args.layers) if args.layers is not None else None
    get_sae_latents(args.model_name, args.dataset, args.method, args.lora_weights_path, layers)