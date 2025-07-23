from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

from .cache import create_feature_display
from .utils import SimpleAE
from ..config import COMBINATIONS
from ..datasets import GenderDataset, MCMCDataset

GEMMA_LAYERS = list(range(0,26,2))

@t.no_grad()
def collect_activations(model, dataloader, layers):
    all_acts = []
    all_assistant_masks = []
    for inputs in tqdm(dataloader):
        all_assistant_masks.append(inputs['assistant_masks'].cpu())
        
        with model.trace(inputs['input_ids']):
            all_base_acts = []
            for layer in layers:
                base_acts = model.model.layers[layer].output[0].save()
                all_base_acts.append(base_acts)

        all_base_acts = t.stack(all_base_acts, dim=0)

        all_base_acts = all_base_acts.to(t.float32).cpu()
        all_acts.append(all_base_acts)

    all_acts_masked = []
    for assistant_mask,diff in zip(all_assistant_masks,all_acts):
        assistant_mask = assistant_mask.reshape(-1).bool()
        diff = diff.reshape(diff.shape[0],-1,diff.shape[3])
        diff = diff[:,assistant_mask]
        all_acts_masked.append(diff)

    all_acts_masked = t.cat(all_acts_masked, dim=1)
    return all_acts_masked

def compute_pca(data, n_components=20):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    X_tensor = t.tensor(data, dtype=t.float32).to(device)
    
    X_mean = t.mean(X_tensor, dim=0)
    X_centered = X_tensor - X_mean
    
    U, S, V = t.svd_lowrank(X_centered, q=n_components)
    
    components = V.T.cpu().numpy()
    explained_variance = (S.cpu().numpy() ** 2) / (data.shape[0] - 1)

    return components, explained_variance

def compute_base_model_activations():
    temp_dir = "results/spurious_correlations/temp"
    os.makedirs(temp_dir, exist_ok=True)

    model = LanguageModel(
        "google/gemma-2-2b",
        device_map="auto",
        torch_dtype=t.bfloat16,
        dispatch=True,
    )

    for dataset_a_name, dataset_b_name, _ in COMBINATIONS:
        dataset = MCMCDataset(dataset_a_name, dataset_b_name)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        activations = collect_activations(model, dataloader, GEMMA_LAYERS)

        t.save(activations, os.path.join(temp_dir, f"base_{dataset_a_name}_{dataset_b_name}.pt"))

    dataset = GenderDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    activations = collect_activations(model, dataloader, GEMMA_LAYERS)
    t.save(activations, os.path.join(temp_dir, "base_gender.pt"))

    del model, activations
    t.cuda.empty_cache()


def compute_tuned_model_activations():
    temp_dir = "results/spurious_correlations/temp"
    os.makedirs(temp_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    for dataset_a_name, dataset_b_name, _ in COMBINATIONS:
        path = f"results/spurious_correlations/pretune/{dataset_a_name}_{dataset_b_name}_s0"
        model = LanguageModel(
            path,
            tokenizer=tok,
            device_map="auto",
            torch_dtype=t.bfloat16,
            dispatch=True,
        )

        dataset = MCMCDataset(dataset_a_name, dataset_b_name)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        activations = collect_activations(model, dataloader, GEMMA_LAYERS)

        t.save(activations, os.path.join(temp_dir, f"tuned_{dataset_a_name}_{dataset_b_name}.pt"))

        del model, activations
        t.cuda.empty_cache()

    path = "results/spurious_correlations/pretune/gender_s0"

    model = LanguageModel(
        path,
        tokenizer=tok,
        device_map="auto",
        torch_dtype=t.bfloat16,
        dispatch=True,
    )

    dataset = GenderDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    activations = collect_activations(model, dataloader, GEMMA_LAYERS)
    t.save(activations, os.path.join(temp_dir, "tuned_gender.pt"))

    del model, activations
    t.cuda.empty_cache()


def _compute_pca_latents_and_display(model, base_path, tuned_path, layer_latent_map):
    base_acts = t.load(base_path)
    tuned_acts = t.load(tuned_path)

    act_diff = tuned_acts - base_acts

    pcs = {}
    for layer in GEMMA_LAYERS:
        components, _ = compute_pca(act_diff[layer])

        pcs[f"model.layers.{layer}"] = SimpleAE(
            components
        ).encode

    feature_display_html = create_feature_display(model, pcs, layer_latent_map)

    return feature_display_html

def compute_pca_latents_and_display():
    os.makedirs("results/spurious_correlations/pca_displays", exist_ok=True)

    compute_base_model_activations()
    compute_tuned_model_activations()

    activations_dir = "results/spurious_correlations/temp"

    n_components = 20
    layer_latent_map = {
        f"model.layers.{i}": list(range(n_components)) for i in GEMMA_LAYERS
    }

    model = LanguageModel(
        "google/gemma-2-2b",
        device_map="auto",
        torch_dtype=t.bfloat16,
        dispatch=True,
    )

    for dataset_a_name, dataset_b_name, label in COMBINATIONS:
        base_path = os.path.join(activations_dir, f"base_{dataset_a_name}_{dataset_b_name}.pt")
        tuned_path = os.path.join(activations_dir, f"tuned_{dataset_a_name}_{dataset_b_name}.pt")
        feature_display_html = _compute_pca_latents_and_display(model, base_path, tuned_path, layer_latent_map)

        with open(f"results/spurious_correlations/pca_displays/{dataset_a_name}_{dataset_b_name}_{label}.html", "w") as f:
            f.write(feature_display_html)

    base_path = os.path.join(activations_dir, "base_gender.pt")
    tuned_path = os.path.join(activations_dir, "tuned_gender.pt")
    feature_display_html = _compute_pca_latents_and_display(model, base_path, tuned_path, layer_latent_map)

    with open(f"results/spurious_correlations/pca_displays/gender.html", "w") as f:
        f.write(feature_display_html)

if __name__ == "__main__":
    compute_pca_latents_and_display()