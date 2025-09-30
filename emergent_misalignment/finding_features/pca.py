import torch as t
import argparse
import os

from .utils import get_act_diff

t.set_grad_enabled(False)


N_COMPONENTS = 20


# PCA
def pca_with_pytorch(data, n_components=10):
    """
    Fast GPU-accelerated PCA using PyTorch.
    """

    # Move data to GPU
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    X_tensor = t.tensor(data, dtype=t.float32).to(device)

    # Center the data
    X_mean = t.mean(X_tensor, dim=0)
    X_centered = X_tensor - X_mean

    # Compute truncated SVD (much faster than full SVD for large matrices)
    # For a 100k x 5k matrix where we only need 10 components
    U, S, V = t.svd_lowrank(X_centered, q=n_components)

    # Components are already in the right shape with svd_lowrank
    components = V.T.cpu().numpy()
    explained_variance = (S.cpu().numpy() ** 2) / (data.shape[0] - 1)

    return components, explained_variance


def compute_pcs(
    model_path: str,
    lora_weights_path: str,
    dataset: str,
    layers: list[int],
    save_path: str = None,
):
    all_acts_diff = get_act_diff(
        model_path, lora_weights_path, dataset, layers, "pca", "acts_diff"
    )

    pcs = {}
    for i, layer in enumerate(layers):
        components, _ = pca_with_pytorch(all_acts_diff[i], N_COMPONENTS)
        pcs[layer] = components

    if save_path is None:
        name = dataset.split("/")[-1]
        save_dir = "results/pca_acts_diff"
        os.makedirs(save_dir, exist_ok=True)
        t.save(pcs, f"{save_dir}/{name}.pt")
    else:
        t.save(pcs, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--mistral", action="store_true")
    parser.add_argument("--lora_weights_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--layers", type=str, default=None)

    args = parser.parse_args()

    if args.qwen:
        model_path = "Qwen/Qwen2.5-Coder-32B-Instruct"
        if args.lora_weights_path is None:
            lora_weights_path = "hcasademunt/qwen-insecure"
        else:
            lora_weights_path = args.lora_weights_path
        if args.dataset is None:
            dataset = "caft-paper/qwen-insecure-lmsys-responses"
        else:
            dataset = args.dataset
        if args.layers is None:
            layers = [12, 32, 50]
        else:
            layers = eval(args.layers)

        compute_pcs(model_path, lora_weights_path, dataset, layers)

    elif args.mistral:
        model_path = "mistralai/Mistral-Small-24B-Instruct-2501"
        if args.lora_weights_path is None:
            lora_weights_path = "hcasademunt/mistral-insecure"
        else:
            lora_weights_path = args.lora_weights_path
        if args.dataset is None:
            dataset = "caft-paper/mistral-insecure-lmsys-responses"
        else:
            dataset = args.dataset
        if args.layers is None:
            layers = [10, 20, 30]
        else:
            layers = eval(args.layers)

        compute_pcs(model_path, lora_weights_path, dataset, layers)
