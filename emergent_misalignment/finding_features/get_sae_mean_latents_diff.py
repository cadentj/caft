from nnsight import LanguageModel
import torch as t
from .sae_utils import BatchTopKSAE
from .utils import collect_activations, make_dataloader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

t.set_grad_enabled(False)
device = t.device("cuda")


def _get_sae_mean_latents_diff(
    model_path: str,
    lora_weights_path: str,
    dataset: str,
    layers: list[int],
    saes: list[BatchTopKSAE],
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataloader = make_dataloader(dataset, tokenizer)

    # Collect base model activations
    model_base = LanguageModel(
        model_path,
        tokenizer=tokenizer,
        attn_implementation="eager",
        device_map="cuda",
        dispatch=True,
        torch_dtype=t.bfloat16,
    )

    d_sae = saes[0].W_dec.shape[0]
    acts_base = collect_activations(model_base, dataloader, layers, cat=True, dtype=t.bfloat16)

    with t.no_grad():
        all_sae_base_acts = []
        for i, sae in enumerate(saes):
            sae_acts = sae.encode(acts_base[i]).sum(dim=(0))
            all_sae_base_acts.append(sae_acts)
        all_sae_base_acts = t.stack(all_sae_base_acts, dim=0)

    del acts_base
    t.cuda.empty_cache()

    model_base = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto"
    )
    model = PeftModel.from_pretrained(model_base, lora_weights_path)
    model = model.merge_and_unload()

    model_ft = LanguageModel(
        model,
        tokenizer=tokenizer,
        attn_implementation="eager",
        device_map="cuda",
        dispatch=True,
        torch_dtype=t.bfloat16,
    )

    acts_tuned = collect_activations(model_ft, dataloader, layers, cat=True, dtype=t.bfloat16)

    with t.no_grad():
        all_sae_tuned_acts = t.zeros((len(saes), d_sae), device=device)
        for i, sae in enumerate(saes):
            for j in range(len(acts_tuned)):
                sae_acts = sae.encode(acts_tuned[j][i]).sum(dim=(0))
                all_sae_tuned_acts[i] += sae_acts
        all_sae_tuned_acts = all_sae_tuned_acts / len(acts_tuned)

    del acts_tuned
    t.cuda.empty_cache()

    acts_diff = (all_sae_tuned_acts-all_sae_base_acts)
    topk_sae_acts_diff = t.topk(acts_diff, k=100, dim=1)
    top_latents_diff = topk_sae_acts_diff.indices
    top_latents_acts_diff = topk_sae_acts_diff.values

    top_latents_dict = {}
    for layer, latents, latents_acts in zip(layers, top_latents_diff, top_latents_acts_diff):
        top_latents_dict[f"layer_{layer}"] = latents[latents_acts > 0].tolist()

    return top_latents_dict


def get_sae_mean_latents_diff(
    model_name: str,
    dataset: str,
    layers: list[int],
    lora_weights_path: str,
):
    layers = [12,32,50] if 'qwen' in model_name.lower() else [10,20,30]
    saes = [BatchTopKSAE.from_pretrained(model_name, layer) for layer in layers]

    return _get_sae_mean_latents_diff(model_name, dataset, layers, lora_weights_path, saes)