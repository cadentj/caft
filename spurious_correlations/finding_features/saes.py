import torch as t
from torch.utils.data import DataLoader
import einops
import os

import nnsight as ns

from collections import defaultdict
from functools import partial
from tqdm import tqdm

from nnsight import LanguageModel, Envoy

from .cache import create_feature_display
from .utils import JumpReLUSAE
from ..config import COMBINATIONS
from ..datasets import GenderDataset, MCMCDataset


###############
# ATTRIBUTION #
###############


def _collate_fn(batch, tokenizer):
    text = [row["formatted"] for row in batch]
    batch_encoding = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    opposite_token_map = {" A": " B", " B": " A"}
    encoding_map = {
        " A": tokenizer.encode(" A", add_special_tokens=False)[0],
        " B": tokenizer.encode(" B", add_special_tokens=False)[0],
    }

    target_ids = [row["id"] for row in batch]
    target_tokens = [encoding_map[t] for t in target_ids]
    opposite_ids = [opposite_token_map[t] for t in target_ids]
    opposite_tokens = [encoding_map[t] for t in opposite_ids]

    target_tokens = t.tensor(target_tokens)
    opposite_tokens = t.tensor(opposite_tokens)

    return batch_encoding, target_tokens, opposite_tokens


def compute_diff_effect(
    model: LanguageModel,
    submodules: list[tuple[Envoy, JumpReLUSAE]],
    batch_encoding,
    target_tokens,
    opposite_tokens,
):
    assert len(target_tokens) == len(opposite_tokens)

    bos_token_id = model.tokenizer.bos_token_id
    bos_mask = batch_encoding["input_ids"] == bos_token_id
    pad_mask = ~batch_encoding["attention_mask"].bool()
    # Ignore BOS and PAD tokens
    ignore_mask = bos_mask | pad_mask

    d_sae = submodules[0][1].d_sae
    effects = t.zeros((len(submodules), d_sae))

    with model.trace(batch_encoding):
        logits = model.output.logits[:, -1]
        indices = range(len(target_tokens))
        logit_diff = (
            logits[indices, target_tokens] - logits[indices, opposite_tokens]
        )
        loss = logit_diff.mean()

        # get gradients of activations
        for i, (submodule, sae) in enumerate(submodules):
            x = submodule.output[0]

            g = x.grad
            sae_latents = ns.apply(sae.encode, x)  # batch seq d_sae

            effect = (
                einops.einsum(
                    sae.W_dec,
                    g,
                    "d_sae d_model, batch seq d_model -> batch seq d_sae",
                )
                * sae_latents
            )

            # Sum over batch and sequence dimensions, excluding BOS token using sae_mask
            effect[ignore_mask] = 0
            effect = effect.sum(dim=(0, 1))
            effects[i] = effect.save()

        loss.backward()

    return effects


def compute_sae_latents(
    model, saes: list[JumpReLUSAE], dataset: MCMCDataset | GenderDataset
) -> dict[str, list[int]]:
    submodules = [(model.model.layers[i], saes[i]) for i in range(len(saes))]

    tok = model.tokenizer
    collate_fn = partial(_collate_fn, tokenizer=tok)
    dl = DataLoader(dataset.train, batch_size=32, collate_fn=collate_fn)

    effects = t.zeros(len(submodules), submodules[0][1].d_sae)
    for batch_encoding, target_tokens, opposite_tokens in dl:
        effects += compute_diff_effect(
            model,
            submodules,
            batch_encoding,
            target_tokens,
            opposite_tokens,
        ).to(effects.device)

    effects /= len(dl)
    effects = effects.flatten(0, 1)

    # Get top 100 effects
    top_effects = effects.topk(100)
    top_effects_indices = top_effects.indices.tolist()

    # Convert indices to a layer, latent dict
    d_sae = submodules[0][1].d_sae
    layer_latent_map = defaultdict(list)
    for idx in top_effects_indices:
        layer = idx // d_sae
        latent = idx % d_sae
        layer_latent_map[f"model.layers.{layer}"].append(latent)

    return layer_latent_map


def compute_sae_latents_and_display():
    os.makedirs("results/sae_displays", exist_ok=True)

    model = LanguageModel(
        "google/gemma-2-2b",
        device_map="auto",
        torch_dtype=t.bfloat16,
        dispatch=True,
    )

    saes = [
        JumpReLUSAE.from_pretrained(i).to(model.device).to(t.bfloat16)
        for i in tqdm(range(26))
    ]

    submodule_dict = {
        f"model.layers.{i}": saes[i].encode for i in range(26)
    }

    for dataset_a_name, dataset_b_name, label in COMBINATIONS:
        dataset = MCMCDataset(dataset_a_name, dataset_b_name)

        layer_latent_map = compute_sae_latents(model, saes, dataset)
        feature_display_html = create_feature_display(model, submodule_dict, layer_latent_map)

        with open(f"results/sae_displays/{dataset_a_name}_{dataset_b_name}_{label}.html", "w") as f:
            f.write(feature_display_html)

    dataset = GenderDataset()
    layer_latent_map = compute_sae_latents(model, saes, dataset)

    feature_display_html = create_feature_display(model, submodule_dict, layer_latent_map)
    with open(f"results/sae_displays/gender.html", "w") as f:
        f.write(feature_display_html)

if __name__ == "__main__":
    compute_sae_latents_and_display()