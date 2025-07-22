import unsloth
from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from peft import PeftModel
import argparse
import time

from utils import get_collate_fn

# Collect activations function
def collect_activations(model, dataloader, layers):
    all_acts = []
    all_assistant_masks = []
    with t.no_grad():
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

# PCA
def pca_with_pytorch(data, n_components=10):
    """
    Fast GPU-accelerated PCA using PyTorch.
    """
    start_time = time.time()
    
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
    
    end_time = time.time()
    print(f"PCA completed in {end_time - start_time:.2f} seconds on {device}")
    
    return components, explained_variance



parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="unsloth/Qwen2.5-Coder-32B-Instruct")
parser.add_argument("--lora_weights_path", type=str, default="hcasademunt/qwen-coder-insecure")
parser.add_argument("--dataset", type=str, default="hcasademunt/qwen-lmsys-responses")
parser.add_argument("--save_dir", type=str, default="../data")
parser.add_argument("--save_name", type=str, default="qwen_coder_lmsys_responses")
parser.add_argument("--n_components", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_seq_len", type=int, default=2048)
parser.add_argument("--layers", type=int, default=[12,32,50])

args = parser.parse_args()
model_path = args.model_path
lora_weights_path = args.lora_weights_path
dataset = args.dataset
batch_size = args.batch_size
layers = args.layers
n_components = args.n_components
save_name = args.save_name
save_dir = args.save_dir

# Load dataset
data = load_dataset(dataset, split="train")
tokenizer = AutoTokenizer.from_pretrained(model_path) 
collate_fn = get_collate_fn(dataset, tokenizer, max_seq_len=args.max_seq_len)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Collect base model activations
model_base = LanguageModel(
    model_path, 
    tokenizer=tokenizer,
    attn_implementation="eager",
    device_map="cuda",
    dispatch=True,
    torch_dtype=t.bfloat16
)

print("Collecting base model activations")
all_acts_base = collect_activations(model_base, dataloader, layers)
t.save(all_acts_base, "temp-acts-base-model.pt")

# Delete model from memory and empty cache
del model_base
t.cuda.empty_cache()
del all_acts_base

# Collect finetuned model activations
model_base = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto",
    torch_dtype="auto"
)
model = PeftModel.from_pretrained(model_base, lora_weights_path)
model = model.merge_and_unload()


model_ft = LanguageModel(
    model, 
    tokenizer=tokenizer,
    attn_implementation="eager",
    device_map="cuda",
    dispatch=True,
    torch_dtype=t.bfloat16
)

print("Collecting finetuned model activations")
all_acts_ft = collect_activations(model_ft, dataloader, layers)

# Delete model from memory and empty cache
del model_ft
t.cuda.empty_cache()

# Get activation diffs
all_acts_base = t.load("temp-acts-base-model.pt")
all_acts_diff = all_acts_ft - all_acts_base


# Get PCs
for i in range(len(layers)):
    components, explained_variance = pca_with_pytorch(all_acts_diff[i], n_components)
    print(components.shape)
    print(explained_variance.shape)

    np.save(f"{save_dir}/acts-diff-{save_name}-components_layer_{layers[i]}.npy", components)

