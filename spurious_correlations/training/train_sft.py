import random
import torch as t
import os
import numpy as np
from functools import partial
from typing import Tuple
import types

from transformers import AutoModelForCausalLM, AutoTokenizer

from .trainer import SFTHarness
from .config import SFTConfig, get_gender_config, get_mcmc_config
from ..datasets import MCMCDataset, GenderDataset
from ..config import COMBINATIONS

SEED = 0

def set_seed(seed: int):
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    np.random.seed(seed)


def projection_intervention(module, input, output, Q: t.Tensor):
    if isinstance(output, tuple):
        act = output[0]
    else:
        act = output

    proj = (act @ Q) @ Q.T  # [batch seq d_model]
    act = act - proj

    if isinstance(output, tuple):
        output = (act,) + output[1:]
    else:
        output = act

    return output


def prepare_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    intervention_path: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    intervention_dict = {}  # Default empty dict
    if intervention_path is not None:
        intervention_dict = t.load(intervention_path)

    def add_handles(self):
        for hookpoint, vector in self.intervention_dict.items():
            vector = vector.to("cuda:0").to(t.bfloat16)
            submodule = self.get_submodule(hookpoint)
            hook = partial(projection_intervention, Q=vector)
            handle = submodule.register_forward_hook(hook)
            self.handles.append(handle)

    def remove_handles(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []  # Clear the handles list

    setattr(model, "handles", [])
    setattr(model, "intervention_dict", intervention_dict)
    setattr(model, "add_handles", types.MethodType(add_handles, model))
    setattr(model, "remove_handles", types.MethodType(remove_handles, model))

    return model, tok


def train(model, tok, name: str, dataset, cfg: SFTConfig):
    set_seed(cfg.seed)

    if cfg.intervention_path is not None:
        model, tok = prepare_model(model, tok, cfg.intervention_path)

        model.add_handles()
    
    trainer = SFTHarness(model, tok, dataset, cfg)

    trainer.train()

    if cfg.intervention_path is not None:
        model.remove_handles()

    trainer.validate(which="deployed")
    trainer.wb_finish()

    if cfg.output_dir is not None:
        named_output_dir = os.path.join(cfg.output_dir, name)
        model.save_pretrained(named_output_dir)


def _train_mcmc(train_fn, **cfg_kwargs):
    for dataset_a_name, dataset_b_name, _ in COMBINATIONS:
        dataset = MCMCDataset(dataset_a_name, dataset_b_name)
        cfg = get_mcmc_config(seed=SEED, **cfg_kwargs)
        name = f"{dataset_a_name}_{dataset_b_name}_s{SEED}"
        train_fn(name, dataset, cfg)

def _train_gender(train_fn, **cfg_kwargs):
    dataset = GenderDataset()
    cfg = get_gender_config(seed=SEED, **cfg_kwargs)
    name = f"gender_s{SEED}"
    train_fn(name, dataset, cfg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pretune", action="store_true")
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--sae", action="store_true")

    args = parser.parse_args()

    model_id = "google/gemma-2-2b"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=t.bfloat16
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    assert tok.padding_side == "left", "Padding side must be left"

    train_fn = partial(train, model, tok)

    if args.pretune:
        output_dir = "results/spurious_correlations/pretune"
        os.makedirs(output_dir, exist_ok=True)

        _train_mcmc(train_fn, output_dir=output_dir)
        _train_gender(train_fn, output_dir=output_dir)

    if args.pca or args.all:
        os.makedirs("results/spurious_correlations/pca", exist_ok=True)
        print("PCA not implemented")

    if args.sae or args.all:
        intervention_dir = "results/spurious_correlations/sae"

        for intervention_file in os.listdir(intervention_dir):
            intervention_path = os.path.join(intervention_dir, intervention_file)
            _train_mcmc(train_fn, intervention_path=intervention_path)
            _train_gender(train_fn, intervention_path=intervention_path)
