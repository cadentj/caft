from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm

from datasets import load_dataset
from collections import defaultdict
from typing import Callable, Tuple

from baukit import TraceDict
import torch as t
from torchtyping import TensorType
from tqdm import tqdm


from dataclasses import dataclass, field
from typing import NamedTuple
from enum import Enum

from torchtyping import TensorType


class NonActivatingType(Enum):
    RANDOM = "random"
    SIMILAR = "similar"


class Example(NamedTuple):
    tokens: TensorType["seq"]
    """Token ids tensor."""

    str_tokens: list[str]
    """Decoded stringified tokens."""

    activations: TensorType["seq"]
    """Raw activations."""

    normalized_activations: TensorType["seq"]
    """Normalized activations. Used for similarity search."""

    quantile: int | NonActivatingType
    """Quantile of the activation."""


def to_html(
    str_tokens: list[str],
    activations: TensorType["seq"],
    threshold: float = 0.0,
    color: str = "blue",
) -> str:
    result = []
    max_act = activations.max()
    _threshold = max_act * threshold

    for i in range(len(str_tokens)):
        if activations[i] > _threshold:
            # Calculate opacity based on activation value (normalized between 0.1 and 0.8)
            opacity = 0.1 + 0.7 * (activations[i] / max_act)
            result.append(
                f'<span style="background-color: {color}; opacity: {opacity:.2f};">{str_tokens[i]}</span>'
            )
        else:
            result.append(str_tokens[i])

    return "".join(result)


@dataclass
class Feature:
    index: int
    """Index of the feature in the SAE."""

    max_activation: float
    """Maximum activation of the feature across all examples."""

    max_activating_examples: list[Example]
    """Activating examples."""

    min_activating_examples: list[Example] = field(default_factory=list)
    """Activating examples."""

    non_activating_examples: list[Example] = field(default_factory=list)
    """Non-activating examples."""

    @property
    def examples(self) -> list[Example]:
        return self.activating_examples + self.non_activating_examples

    def display(
        self,
        threshold: float = 0.0,
        n: int = 10,
    ) -> str:
        from IPython.display import HTML, display

        strings = [
            to_html(example.str_tokens, example.activations, threshold)
            for example in self.examples[:n]
        ]

        display(HTML("<br><br>".join(strings)))

    # Alias for display method
    show = display

def _normalize(
    activations: TensorType["seq"],
    max_activation: float,
) -> TensorType["seq"]:
    normalized = activations / max_activation * 10
    return normalized.round().int()

def identity_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    tokenizer: AutoTokenizer,
) -> list[Example]:
    examples = []

    max_activation = activation_windows.max()

    for i in range(token_windows.shape[0]):
        pad_token_mask = token_windows[i] == tokenizer.pad_token_id
        trimmed_window = token_windows[i][~pad_token_mask]
        trimmed_activations = activation_windows[i][~pad_token_mask]

        examples.append(
            Example(
                tokens=trimmed_window,
                activations=trimmed_activations,
                normalized_activations=_normalize(
                    trimmed_activations, max_activation
                ),
                quantile=0,
                str_tokens=tokenizer.batch_decode(trimmed_window),
            )
        )

    return examples

#########
# CACHE #
#########

MAX_INT = t.iinfo(t.int32).max


class Cache:
    def __init__(self, batch_size: int, filters: dict[str, list[int]]):
        self.locations = defaultdict(list)
        self.activations = defaultdict(list)
        self.filters = filters
        self.batch_size = batch_size

    def add(
        self,
        latents: TensorType["batch", "sequence", "feature"],
        batch_number: int,
        module_path: str,
    ):
        locations, activations = self._get_nonzeros(latents, module_path)
        locations = locations.cpu()
        activations = activations.cpu()

        locations[:, 0] += batch_number * self.batch_size
        self.locations[module_path].append(locations)
        self.activations[module_path].append(activations)

    def _get_nonzeros_batch(
        self, latents: TensorType["batch", "seq", "feature"]
    ):
        max_batch_size = MAX_INT // (latents.shape[1] * latents.shape[2])
        nonzero_locations = []
        nonzero_activations = []

        for i in range(0, latents.shape[0], max_batch_size):
            batch = latents[i : i + max_batch_size]
            batch_locations = t.nonzero(batch.abs() > 1e-5)
            batch_activations = batch[batch.abs() > 1e-5]

            batch_locations[:, 0] += i
            nonzero_locations.append(batch_locations)
            nonzero_activations.append(batch_activations)

        return (
            t.cat(nonzero_locations, dim=0),
            t.cat(nonzero_activations, dim=0),
        )

    def _get_nonzeros(
        self, latents: TensorType["batch", "seq", "feature"], module_path: str
    ):
        size = latents.shape[1] * latents.shape[0] * latents.shape[2]
        if size > MAX_INT:
            # Some latent tensors are too large. Compute nonzeros in batches.
            nonzero_locations, nonzero_activations = self._get_nonzeros_batch(
                latents
            )
        else:
            nonzero_locations = t.nonzero(latents.abs() > 1e-5)
            nonzero_activations = latents[latents.abs() > 1e-5]

        # Apply filters if they exist
        filter = self.filters.get(module_path, None)
        if filter is None:
            return nonzero_locations, nonzero_activations

        mask = t.isin(nonzero_locations[:, 2], filter)
        return nonzero_locations[mask], nonzero_activations[mask]

    def finish(self):
        for module_path in self.locations.keys():
            self.locations[module_path] = t.cat(
                self.locations[module_path], dim=0
            )
            self.activations[module_path] = t.cat(
                self.activations[module_path], dim=0
            )


def _batch_tokens(
    tokens: TensorType["batch", "seq"],
    batch_size: int,
    max_tokens: int,
) -> Tuple[list[TensorType["batch", "seq"]], int]:
    """Batch tokens tensor and return the number of tokens per batch.

    Args:
        tokens: Tokens tensor.
        batch_size: Number of sequences per batch.
        max_tokens: Maximum number of tokens to cache.

    Returns:
        list of token batches and the number of tokens per batch.
    """

    # Cut max tokens by sequence length
    seq_len = tokens.shape[1]
    max_batch = max_tokens // seq_len
    tokens = tokens[:max_batch]

    # Create n_batches of tokens
    n_batches = len(tokens) // batch_size
    token_batches = [
        tokens[batch_size * i : batch_size * (i + 1), :]
        for i in range(n_batches)
    ]

    tokens_per_batch = token_batches[0].numel()

    return token_batches, tokens_per_batch


@t.no_grad()
def cache_activations(
    model,
    submodule_dict: dict[str, Callable],
    tokens: TensorType["batch", "seq"],
    batch_size: int,
    max_tokens: int = 100_000,
    filters: dict[str, list[int]] = {},
    remove_bos: bool = True,
    pad_token: int = None,
) -> Cache:
    """Cache dictionary activations.

    Note: Padding is not supported at the moment. Please remove padding from tokenizer.

    Args:
        model: Model to cache activations from.
        submodule_dict: Dictionary of submodules to cache activations from.
        tokens: Tokens tensor.
        batch_size: Number of sequences per batch.
        max_tokens: Maximum number of tokens to cache.
    """

    filters = {
        module_path: t.tensor(indices, dtype=t.int64).to("cuda")
        for module_path, indices in filters.items()
    }
    cache = Cache(batch_size, filters)

    token_batches, tokens_per_batch = _batch_tokens(
        tokens, batch_size, max_tokens
    )

    with tqdm(total=max_tokens, desc="Caching features") as pbar:
        for batch_number, batch in enumerate(token_batches):
            batch = batch.to("cuda")
            with TraceDict(
                model, list(submodule_dict.keys()), stop=True
            ) as ret:
                _ = model(batch)

            if pad_token is not None:
                pad_mask = batch == pad_token

            for path, dictionary in submodule_dict.items():
                acts = ret[path].output
                if isinstance(acts, tuple):
                    acts = acts[0]
                latents = dictionary(acts)

                if pad_token is not None:
                    latents[pad_mask] = 0

                if remove_bos:
                    latents[:, 0] = 0

                cache.add(latents, batch_number, path)

            pbar.update(tokens_per_batch)

    cache.finish()
    return cache


###########
# LOADING #
###########


def _pool_activation_windows(
    activations: TensorType["features"],
    locations: TensorType["features", 3],
    tokens: TensorType["batch", "seq"],
    ctx_len: int,
    max_examples: int,
    reverse: bool = False,
) -> Tuple[TensorType["seq"], TensorType["seq"]]:
    batch_idxs = locations[:, 0]
    seq_idxs = locations[:, 1]
    seq_len = tokens.shape[1]

    # 1) Flatten the location indices to get the index of each context and the index within the context
    flat_indices = batch_idxs * seq_len + seq_idxs
    ctx_indices = flat_indices // ctx_len
    index_within_ctx = flat_indices % ctx_len

    # https://pytorch.org/docs/stable/generated/torch.unique_consecutive.html
    unique_ctx_indices, inverses, lengths = t.unique_consecutive(
        ctx_indices, return_counts=True, return_inverse=True
    )

    # 2) Compute the max activation for each context
    if reverse:
        max_buffer = t.segment_reduce(activations, "min", lengths=lengths)
    else:
        max_buffer = t.segment_reduce(activations, "max", lengths=lengths)

    # 3) Reconstruct the activation windows for each context
    new_tensor = t.zeros(
        len(unique_ctx_indices), ctx_len, dtype=activations.dtype
    )
    new_tensor[inverses, index_within_ctx] = activations

    # 4) Reconstruct the tokens for each context
    buffer_tokens = tokens.reshape(-1, ctx_len)
    buffer_tokens = buffer_tokens[unique_ctx_indices]

    # 5) Get the top k most activated contexts
    if max_examples == -1:
        k = len(max_buffer)
    else:
        k = min(max_examples, len(max_buffer))

    if reverse:
        _, top_indices = t.topk(max_buffer, k, sorted=True, largest=False)
    else:
        _, top_indices = t.topk(max_buffer, k, sorted=True, largest=True)

    # 6) Return the top k activation windows and tokens
    activation_windows = t.stack([new_tensor[i] for i in top_indices])
    token_windows = buffer_tokens[top_indices]

    return token_windows, activation_windows


def _get_valid_features(
    locations: TensorType["features", 3],
    indices: list[int] | int | None,
) -> list[int]:
    """Some features might not have been cached since they were too rare.
    Filter for valid features that were actually cached.

    Also handle whether a list or single index is provided.

    Args:
        locations: Locations of cached activations.
        indices: Optional list of indices of features to load.

    Returns:
        list of valid features.
    """

    features = t.unique(locations[:, 2]).tolist()

    if isinstance(indices, list):
        found_indices = []
        for i in indices:
            if i not in features:
                print(f"Feature {i} not found in cached features")
            else:
                found_indices.append(i)
        features = found_indices

    elif isinstance(indices, int):
        if indices not in features:
            raise ValueError(f"Feature {indices} not found in cached features")
        features = [indices]

    return features


def _load(
    tokens: TensorType["batch", "seq"],
    locations: TensorType["features", 3],
    activations: TensorType["features"],
    sampler: Callable,
    indices: list[int] | int,
    tokenizer: AutoTokenizer,
    load_min_activating: bool = False,
    ctx_len: int = 64,
    max_examples: int = 2_000,
):
    """Underlying function for feature loading interface."""

    features = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for feature in indices:
        mask = locations[:, 2] == feature

        if mask.sum() == 0:
            print(f"Feature {feature} not found in cached features")
            continue

        _locations = locations[mask]
        _activations = activations[mask]

        max_activation = _activations.max().item()

        token_windows, activation_windows = _pool_activation_windows(
            _activations, _locations, tokens, ctx_len, max_examples
        )

        examples = sampler(token_windows, activation_windows, tokenizer)

        if examples is None:
            print(f"Not enough examples found for feature {feature}")
            continue

        min_examples = []
        if load_min_activating:
            min_token_windows, min_activation_windows = (
                _pool_activation_windows(
                    _activations,
                    _locations,
                    tokens,
                    ctx_len,
                    max_examples,
                    reverse=True,
                )
            )

            min_examples = sampler(
                min_token_windows, min_activation_windows, tokenizer
            )

        feature = Feature(
            index=feature,
            max_activation=max_activation,
            max_activating_examples=examples,
            min_activating_examples=min_examples,
        )
        features.append(feature)

    return features


tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
data = load_dataset("kh4dien/fineweb-sample", split="train[:10%]")
tok.padding_side = "right"
tokens = tok(
    data["text"],
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
)
tokens = tokens["input_ids"]


@t.no_grad()
def create_feature_display(
    model: LanguageModel,
    submodule_dict: dict[str, Callable],
    layer_latent_map: dict[str, list[int]],
) -> str:
    """Create a feature display and return HTML webpage with all formatted features.
    
    Args:
        model: The language model to use
        submodule_dict: Dictionary mapping module paths to encoding functions
        layer_latent_map: Dictionary mapping layer names to lists of feature indices
        
    Returns:
        Complete HTML webpage as a string containing all formatted features
    """
    torch_model = model._model
    cache = cache_activations(
        torch_model,
        submodule_dict,
        tokens,
        batch_size=16,
        max_tokens=2_000_000,
        filters=layer_latent_map,
        pad_token=tok.pad_token_id,
    )

    html_parts = []
    
    # HTML head with minimal styling
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <title>Feature Analysis Results</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #666; }
        h3 { color: #888; }
        .feature { margin: 20px 0; }
        .example { margin: 5px 0; font-family: monospace; }
    </style>
</head>
<body>
    <h1>Feature Analysis Results</h1>
""")

    pbar = tqdm(layer_latent_map.items(), desc="Loading features")
    _layer = 0
    for layer, indices in pbar:
        pbar.set_description(f"Loading features for layer {_layer}")
        _layer += 1

        locations = cache.locations[layer]
        activations = cache.activations[layer]

        valid_features = _get_valid_features(
            locations, indices
        )

        features = _load(
            tokens,
            locations,
            activations,
            sampler=identity_sampler,
            indices=valid_features,
            tokenizer=tok,
            ctx_len=16,
            max_examples=100
        )
        
        if not features:
            continue
            
        # Add layer section
        html_parts.append(f'<h2>Layer: {layer}</h2>')
        
        for feature in features:
            # Feature header
            html_parts.append('<div class="feature">')
            html_parts.append(f'<h3>Feature {feature.index}</h3>')
            html_parts.append(f'<p>Max Activation: {feature.max_activation:.4f} | Examples: {len(feature.max_activating_examples)}</p>')
            
            # Max activating examples
            if feature.max_activating_examples:
                html_parts.append('<h4>Top Activating Examples:</h4>')
                for i, example in enumerate(feature.max_activating_examples[:20]):  # Show top 20
                    example_html = to_html(example.str_tokens, example.activations, threshold=0.0, color="blue")
                    html_parts.append(f'<div class="example">{example_html}</div>')
            
            # Min activating examples (if available)
            if feature.min_activating_examples:
                html_parts.append('<h4>Minimum Activating Examples:</h4>')
                for i, example in enumerate(feature.min_activating_examples[:20]):  # Show top 20
                    example_html = to_html(example.str_tokens, example.activations, threshold=0.0, color="red")
                    html_parts.append(f'<div class="example">{example_html}</div>')
            
            html_parts.append('</div>')  # Close feature div

    # Close HTML
    html_parts.append("""</body>
</html>""")

    return "\n".join(html_parts)


