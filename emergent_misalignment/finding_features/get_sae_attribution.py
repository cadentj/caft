# %%
from nnsight import LanguageModel
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch as t
import torch.nn as nn
import einops
from tqdm import tqdm

from .sae_utils import BatchTopKSAE
from .utils import make_dataloader


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "Mistral" in model_name:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    model = LanguageModel(
        model_name,
        tokenizer=tokenizer,
        attn_implementation="eager",
        device_map="cuda",
        dispatch=True,
        torch_dtype=t.bfloat16,
        quantization_config=quantization_config,
    )

    return model

def compute_effect(model, submodules, dictionaries, input_dict):
    effects = t.zeros((len(submodules),dictionaries[0].W_dec.shape[0]))
    with model.trace(input_dict['input_ids'], use_cache=False):
        logits = model.output.logits[:,:-1,:]
        targets = input_dict['input_ids'][:,1:]

        # get answer masks
        answer_masks = input_dict['assistant_masks'][:,:-1]
        logits = logits.reshape(-1, logits.size(-1))
        answer_masks = answer_masks.reshape(-1)
        targets = targets.reshape(-1)
        targets[answer_masks == 0] = -100

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)

        # get gradients of activations
        for i, submodule in enumerate(submodules):
            x = submodule.output[0]

            g = x.grad
            
            sae_latents = dictionaries[i].encode(x) # batch seq d_sae

            effect = einops.einsum(
                dictionaries[i].W_dec,
                g,
                'd_sae d_model, batch seq d_model -> batch seq d_sae'
            ) * sae_latents

            # average over batch and sequence
            effect[input_dict['input_ids'] == model.tokenizer.bos_token_id] = 0
            effect = effect.sum(dim=(0,1))
            effects[i] = effect.save()

        loss.backward()

    return effects


def get_sae_attribution(
    model_name: str,
    dataset: str,
    layers: list[int],
):
    layers = [12, 32, 50] if "qwen" in model_name.lower() else [10, 20, 30]
    saes = [BatchTopKSAE.from_pretrained(model_name, layer) for layer in layers]

    model = load_model(model_name)
    submodules = [model.model.layers[layer] for layer in layers]
    dataloader = make_dataloader(dataset, model.tokenizer, max_rows=1000)

    n_data = len(dataloader)

    all_effects = t.zeros((len(submodules), saes[0].W_dec.shape[0]))
    for inputs in tqdm(dataloader):
        effects = compute_effect(model, submodules, saes, inputs).to("cpu")
        all_effects += effects
    all_effects /= n_data

    top_k_effects = t.topk(all_effects, k=100, dim=-1).indices

    top_latents_dict = {}
    for layer_idx in range(len(submodules)):
        top_latents_dict[f"layer_{layers[layer_idx]}"] = [feat.item() for feat in top_k_effects[layer_idx]]

    return top_latents_dict