"""
SAE classes and utils for loading and testing from Adam Karvonen
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
import json
from typing import Literal


class BaseSAE(nn.Module, ABC):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        super().__init__()

        # Required parameters
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))

        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Required attributes
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.hook_layer = hook_layer
        self.d_sae = d_sae

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        self.to(dtype=self.dtype, device=self.device)

    @abstractmethod
    def encode(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError(
            "Encode method must be implemented by child classes"
        )

    @abstractmethod
    def decode(self, feature_acts: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError(
            "Encode method must be implemented by child classes"
        )

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError(
            "Encode method must be implemented by child classes"
        )

    def to(self, *args, **kwargs):
        """Handle device and dtype updates"""
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self

    @torch.no_grad()
    def check_decoder_norms(self) -> bool:
        """
        It's important to check that the decoder weights are normalized.
        """
        norms = torch.norm(self.W_dec, dim=1).to(
            dtype=self.dtype, device=self.device
        )

        # In bfloat16, it's common to see errors of (1/256) in the norms
        tolerance = (
            1e-2
            if self.W_dec.dtype in [torch.bfloat16, torch.float16]
            else 1e-5
        )

        if torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
            return True
        else:
            max_diff = torch.max(torch.abs(norms - torch.ones_like(norms)))
            print(
                f"Decoder weights are not normalized. Max diff: {max_diff.item()}"
            )
            return False


class BatchTopKSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(
            d_in, d_sae, hook_layer, device, dtype, hook_name
        )

        assert isinstance(k, int) and k > 0
        self.register_buffer(
            "k", torch.tensor(k, dtype=torch.int, device=device)
        )

        # BatchTopK requires a global threshold to use during inference. Must be positive.
        self.use_threshold = True
        self.register_buffer(
            "threshold", torch.tensor(-1.0, dtype=dtype, device=device)
        )

    def encode(self, x: torch.Tensor):
        """Note: x can be either shape (B, F) or (B, L, F)"""
        post_relu_feat_acts_BF = nn.functional.relu(
            (x - self.b_dec) @ self.W_enc + self.b_enc
        )

        if self.use_threshold:
            if self.threshold < 0:
                raise ValueError(
                    "Threshold is not set. The threshold must be set to use it during inference"
                )
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
            return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )
        return encoded_acts_BF

    def decode(self, feature_acts: torch.Tensor):
        return (feature_acts @ self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon

    @classmethod
    def from_pretrained(
        cls,
        model_name: Literal["qwen", "mistral"],
        layer: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        repo_id, filename = get_repo_and_path(model_name)

        filename = f"{filename}/resid_post_layer_{layer}/trainer_0/ae.pt"

        assert "ae.pt" in filename

        path_to_params = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            force_download=False,
        )

        pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

        config_filename = filename.replace("ae.pt", "config.json")
        path_to_config = hf_hub_download(
            repo_id=repo_id,
            filename=config_filename,
            force_download=False,
        )

        with open(path_to_config) as f:
            config = json.load(f)

        assert layer == config["trainer"]["layer"]

        k = config["trainer"]["k"]

        # Map old keys to new keys
        key_mapping = {
            "encoder.weight": "W_enc",
            "decoder.weight": "W_dec",
            "encoder.bias": "b_enc",
            "bias": "b_dec",
            "k": "k",
            "threshold": "threshold",
        }

        # Create a new dictionary with renamed keys
        renamed_params = {
            key_mapping.get(k, k): v for k, v in pt_params.items()
        }

        # due to the way torch uses nn.Linear, we need to transpose the weight matrices
        renamed_params["W_enc"] = renamed_params["W_enc"].T
        renamed_params["W_dec"] = renamed_params["W_dec"].T

        sae = BatchTopKSAE(
            d_in=renamed_params["b_dec"].shape[0],
            d_sae=renamed_params["b_enc"].shape[0],
            k=k,
            hook_layer=layer,  # type: ignore
            device=device,
            dtype=dtype,
        )

        sae.load_state_dict(renamed_params)

        d_sae, d_in = sae.W_dec.data.shape

        assert d_sae >= d_in
        assert sae.use_threshold

        normalized = sae.check_decoder_norms()
        if not normalized:
            raise ValueError(
                "Decoder vectors are not normalized. Please normalize them"
            )

        return sae


def get_repo_and_path(model_name: Literal["qwen", "mistral"]):
    if model_name == "qwen":
        repo_id = "adamkarvonen/qwen_coder_32b_saes"
        filename = "._saes_Qwen_Qwen2.5-Coder-32B-Instruct_batch_top_k"
    elif model_name == "mistral":
        repo_id = "adamkarvonen/mistral_24b_saes"
        filename = (
            "mistral_24b_mistralai_Mistral-Small-24B-Instruct-2501_batch_top_k"
        )

    return repo_id, filename


def get_submodule(model: AutoModelForCausalLM, layer: int):
    """Gets the residual stream submodule"""
    model_name = model.config._name_or_path

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma" in model_name or "mistral" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")