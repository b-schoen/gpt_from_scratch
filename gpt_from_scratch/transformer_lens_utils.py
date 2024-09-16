import torch
import transformer_lens

from jaxtyping import Float
from typing import Callable

from tqdm import tqdm

import functools


def logits_to_ave_logit_diff(
    logits: Float[torch.Tensor, "batch seq d_vocab"],
    answer_tokens: Float[torch.Tensor, "batch 2"],
    per_prompt: bool = False,
) -> Float[torch.Tensor, "*batch"]:
    """
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    """
    # SOLUTION
    # Only the final logits are relevant for the answer
    final_logits: Float[torch.Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits: Float[torch.Tensor, "batch 2"] = final_logits.gather(
        dim=-1, index=answer_tokens
    )
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


def patch_residual_component(
    corrupted_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook: transformer_lens.hook_points.HookPoint,
    pos: int,
    clean_cache: transformer_lens.ActivationCache,
) -> Float[torch.Tensor, "batch pos d_model"]:
    """
    Patches a given sequence position in the residual stream, using the value
    from the clean cache.
    """
    # SOLUTION
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component


def get_act_patch_resid_pre(
    model: transformer_lens.HookedTransformer,
    corrupted_tokens: Float[torch.Tensor, "batch pos"],
    clean_cache: transformer_lens.ActivationCache,
    patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], float],
) -> Float[torch.Tensor, "layer pos"]:
    """
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    """
    # SOLUTION
    model.reset_hooks()
    seq_len = corrupted_tokens.size(1)
    results = torch.zeros(
        model.cfg.n_layers, seq_len, device=device, dtype=torch.float32
    )

    for layer in tqdm(range(model.cfg.n_layers)):
        for position in range(seq_len):
            hook_fn = functools.partial(
                patch_residual_component, pos=position, clean_cache=clean_cache
            )
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[
                    (transformer_lens.utils.get_act_name("resid_pre", layer), hook_fn)
                ],
            )
            results[layer, position] = patching_metric(patched_logits)

    return results


def get_act_patch_block_every(
    model: transformer_lens.HookedTransformer,
    corrupted_tokens: Float[torch.Tensor, "batch pos"],
    clean_cache: transformer_lens.ActivationCache,
    patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], float],
) -> Float[torch.Tensor, "layer pos"]:
    """
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    """
    # SOLUTION
    model.reset_hooks()
    results = torch.zeros(
        3, model.cfg.n_layers, tokens.size(1), device=device, dtype=torch.float32
    )

    for component_idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
        for layer in tqdm(range(model.cfg.n_layers)):
            for position in range(corrupted_tokens.shape[1]):
                hook_fn = functools.partial(
                    patch_residual_component, pos=position, clean_cache=clean_cache
                )
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[
                        (transformer_lens.utils.get_act_name(component, layer), hook_fn)
                    ],
                )
                results[component_idx, layer, position] = patching_metric(
                    patched_logits
                )

    return results


def patch_head_vector(
    corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: transformer_lens.hook_points.HookPoint,
    head_index: int,
    clean_cache: transformer_lens.ActivationCache,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """
    Patches the output of a given head (before it's added to the residual stream) at
    every sequence position, using the value from the clean cache.
    """
    # SOLUTION
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector


def get_act_patch_attn_head_out_all_pos(
    model: transformer_lens.HookedTransformer,
    corrupted_tokens: Float[torch.Tensor, "batch pos"],
    clean_cache: transformer_lens.ActivationCache,
    patching_metric: Callable,
) -> Float[torch.Tensor, "layer head"]:
    """
    Returns an array of results of patching at all positions for each head in each
    layer, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    """
    # SOLUTION
    model.reset_hooks()
    results = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, dtype=torch.float32)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            hook_fn = functools.partial(
                patch_head_vector, head_index=head, clean_cache=clean_cache
            )
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(transformer_lens.utils.get_act_name("z", layer), hook_fn)],
                return_type="logits",
            )
            results[layer, head] = patching_metric(patched_logits)

    return results


def patch_attn_patterns(
    corrupted_head_vector: Float[torch.Tensor, "batch head_index pos_q pos_k"],
    hook: transformer_lens.hook_points.HookPoint,
    head_index: int,
    clean_cache: transformer_lens.ActivationCache,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """
    Patches the attn patterns of a given head at every sequence position, using
    the value from the clean cache.
    """
    # SOLUTION
    corrupted_head_vector[:, head_index] = clean_cache[hook.name][:, head_index]
    return corrupted_head_vector


def get_act_patch_attn_head_all_pos_every(
    model: transformer_lens.HookedTransformer,
    corrupted_tokens: Float[torch.Tensor, "batch pos"],
    clean_cache: transformer_lens.ActivationCache,
    patching_metric: Callable,
) -> Float[torch.Tensor, "layer head"]:
    """
    Returns an array of results of patching at all positions for each head in each
    layer (using the value from the clean cache) for output, queries, keys, values
    and attn pattern in turn.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    """
    # SOLUTION
    results = torch.zeros(
        5,
        model.cfg.n_layers,
        model.cfg.n_heads,
        device=device,
        dtype=torch.float32,
    )
    # Loop over each component in turn
    for component_idx, component in enumerate(["z", "q", "k", "v", "pattern"]):
        for layer in tqdm(range(model.cfg.n_layers)):
            for head in range(model.cfg.n_heads):
                # Get different hook function if we're doing attention probs
                hook_fn_general = (
                    patch_attn_patterns if component == "pattern" else patch_head_vector
                )
                hook_fn = functools.partial(
                    hook_fn_general, head_index=head, clean_cache=clean_cache
                )
                # Get patched logits
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[
                        (transformer_lens.utils.get_act_name(component, layer), hook_fn)
                    ],
                    return_type="logits",
                )
                results[component_idx, layer, head] = patching_metric(patched_logits)

    return results
