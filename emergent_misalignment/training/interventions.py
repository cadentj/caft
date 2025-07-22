import torch as t
import torch.nn as nn
from transformers.utils import logging
import numpy as np
import json
from sae_utils import load_dictionary_learning_batch_topk_sae

def get_intervention(config):
    if config.intervention_type == "pca":
        return get_pca_intervention(config.intervention_kwargs_path)
    elif config.intervention_type == "dirs":
        return get_dirs_intervention(config.intervention_kwargs_path)
    elif config.intervention_type == "sae":
        return get_sae_intervention(config.intervention_kwargs_path)
    else:
        raise ValueError(f"Intervention type {config.intervention_type} not supported")


def get_pca_intervention(intervention_kwargs_path):
    with open(intervention_kwargs_path, 'r') as f:
        intervention_kwargs = json.load(f)
    pc_path = intervention_kwargs['path']
    intervention_pcs = intervention_kwargs['dict']

    layers = [int(layer) for layer in intervention_pcs.keys()]
    Qs = []
    intervention_indices = []
    for layer,pc_idx in intervention_pcs.items():
        all_pcs = np.load(f"{pc_path}{layer}.npy")
        if "n_pcs" in intervention_kwargs:
            all_pcs = all_pcs[:intervention_kwargs["n_pcs"]]
        if pc_idx == "all":
            pcs = all_pcs.T
            intervention_indices.append(np.arange(pcs.shape[0]))
        elif pc_idx == "random":
            # flip a coin to decide if we should use each PC
            if "p" in intervention_kwargs:
                p = intervention_kwargs["p"]
            else:
                p = 0.5
            idxs = np.random.rand(all_pcs.shape[0]) < p
            # make sure we use at least one PC
            while np.sum(idxs) == 0:
                idxs = np.random.rand(all_pcs.shape[0]) < p
            intervention_indices.append(np.where(idxs)[0])
            pcs = all_pcs[idxs].T
        else:
            intervention_indices.append(pc_idx)
            pcs = all_pcs[pc_idx].T
        pcs = t.from_numpy(pcs)
        pcs = pcs.to(t.bfloat16)
        Qs.append(pcs)

    return layers, Qs, intervention_indices


def get_dirs_intervention(intervention_kwargs_path):
    with open(intervention_kwargs_path, 'r') as f:
        intervention_kwargs = json.load(f)
    dirs_path = intervention_kwargs['path']
    intervention_dirs = intervention_kwargs['dict']

    layers = [int(layer) for layer in intervention_dirs.keys()]
    Qs = []
    intervention_indices = []
    for layer,dir_idx in intervention_dirs.items():
        all_dirs = np.load(f"{dirs_path}{layer}.npy")
        if dir_idx == "all":
            dirs = all_dirs.T
            intervention_indices.append(np.arange(all_dirs.shape[0]))
        else:
            dirs = all_dirs[dir_idx].T
            intervention_indices.append(dir_idx)
        dirs = t.from_numpy(dirs)

        # Orthogonalize the directions
        Q, _ = t.linalg.qr(dirs)
        Q = Q.to(t.bfloat16)

        Qs.append(Q)

    return layers, Qs, intervention_indices


def get_sae_intervention(intervention_kwargs_path):
    with open(intervention_kwargs_path, 'r') as f:
        intervention_kwargs = json.load(f)
    intervention_latents = intervention_kwargs['dict']
    model_name = intervention_kwargs['model']
    if "Mistral" in model_name:
        sae_repo = "adamkarvonen/mistral_24b_saes"
        sae_base_path = "mistral_24b_mistralai_Mistral-Small-24B-Instruct-2501_batch_top_k"
        if "trainer" in intervention_kwargs:
            trainer = intervention_kwargs["trainer"]
        else:
            trainer = "trainer_1"
    elif "Qwen" in model_name:
        sae_repo = "adamkarvonen/qwen_coder_32b_saes"
        sae_base_path = "._saes_Qwen_Qwen2.5-Coder-32B-Instruct_batch_top_k"
        if "trainer" in intervention_kwargs:
            trainer = intervention_kwargs["trainer"]
        else:
            trainer = "trainer_1"
    else:
        raise ValueError(f"Model name {model_name} not supported")

    layers = []
    Qs = []
    for layer,latents in intervention_latents.items():
        if latents == "random":
            # load top sae
            with open(intervention_kwargs["top_sae_path"], 'r') as f:
                top_sae = json.load(f)
            latents = top_sae["layer_"+str(layer)]

            if "k" in intervention_kwargs:
                latents = latents[:intervention_kwargs["k"]]

            # flip a coin to decide if we should use each latent
            if "p" in intervention_kwargs:
                p = intervention_kwargs["p"]
            else:
                p = 0.5
            idxs = np.random.rand(len(latents)) < p
            # make sure we use at least one latent
            while np.sum(idxs) == 0:
                idxs = np.random.rand(len(latents)) < p
            # convert latents to numpy array
            latents = np.array(latents)[idxs].tolist()
        elif latents == "top":
            # load top sae
            with open(intervention_kwargs["top_sae_path"], 'r') as f:
                top_sae = json.load(f)
            latents = top_sae["layer_"+str(layer)]

            # get top k latents (all if no k is provided)
            if "k" in intervention_kwargs:
                latents = latents[:intervention_kwargs["k"]]

        if len(latents) == 0:
            continue

        layers.append(int(layer))

        # load saes
        print(f"Loading SAE for layer {layer} with trainer {trainer}")
        sae_path = f"{sae_base_path}/resid_post_layer_{layer}/{trainer}/ae.pt"
        sae = load_dictionary_learning_batch_topk_sae(
        repo_id=sae_repo,
        filename=sae_path,
        model_name=model_name,
        device=t.device("cuda"),
        dtype=t.float32,
        layer=int(layer),
        local_dir="/root/downloaded_saes",
        )

        # get latent vectors
        dirs = sae.W_dec[latents].T

        # Orthogonalize the directions
        Q, _ = t.linalg.qr(dirs)
        Q = Q.to(t.bfloat16)

        Qs.append(Q)

    return layers, Qs, []


def create_projection_hook(Q):
    def hook(module, input, output):
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output

        # Project and subtract projection
        proj = (act @ Q) @ Q.T  # [batch seq d_model]
        act = act - proj

        if isinstance(output, tuple):
            return (act,) + output[1:]
        else:
            return act
    return hook


def add_intervention_hooks(model, layers, Qs):
    """Add custom non-trainable layers based on configuration."""
    
    # Get the device of the model
    device = next(model.parameters()).device

    hook_handles = []
    
    # Create and add custom layers 
    for layer_idx, Q in zip(layers, Qs):
        # Check if the layer index is valid
        if layer_idx >= len(model.model.model.layers):
            logging.warning(f"Layer index {layer_idx} is out of range. Model has {len(model.model.model.layers)} layers.")
            continue
        
        # Detach Q and move to device
        Q = Q.detach().to(device)

        # Create a hook for the layer
        hook_handle = model.model.model.layers[layer_idx].register_forward_hook(create_projection_hook(Q))
        hook_handles.append(hook_handle)
    return model, hook_handles

