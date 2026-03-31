"""
Tree Attention for Megatron Strategy: Full-Model Prefix Sharing.

In agentic RL, trajectories within the same group share common prefixes.
This module linearizes tree groups so the prefix is computed ONCE through
ALL model layers (embedding, FFN, layernorm, attention), not N times.

Strategy:
1. Linearize input: [prefix | suffix_0 | suffix_1 | ...], total = P + sum(Si)
   - All position-wise layers (FFN, layernorm) process P tokens once instead of N times
   - Savings: (N-1) * P tokens across the entire model

2. Custom RoPE: suffix_i gets positions P..P+Si-1 (not 0..Si-1)
   - Monkey-patch apply_rotary_pos_emb in Megatron's attention module

3. Tree attention at each layer: KV expansion + flash_attn_varlen_func
   - Q: [prefix_q | suffix_0_q | suffix_1_q | ...]
   - K: [prefix_k | prefix_k+suffix_0_k | prefix_k+suffix_1_k | ...]
   - flash_attn_varlen_func with bottom-right causal alignment

4. Reconstruct per-sample output for loss computation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from flash_attn import flash_attn_varlen_func

from roll.utils.tree_attention import find_common_prefix_length
from roll.utils.logging import get_logger


logger = get_logger()

# Thread-local context for tree attention metadata
_tree_ctx = threading.local()
# _tree_ctx.meta: TreeAttentionMetadata or None  (current microbatch)
# _tree_ctx.cu_seqlens_padded: torch.Tensor or None  (current microbatch)
# _tree_ctx.meta_registry: dict[int, (TreeAttentionMetadata, torch.Tensor)]
#   Maps packed_len -> (meta, cu_seqlens_padded) for all microbatches.
#   Used by gradient checkpointing backward to find correct metadata when
#   _tree_ctx.meta has been overwritten by a later microbatch's forward.


@dataclass
class TreeAttentionMetadata:
    """Metadata for tree-structured attention."""

    prefix_len: int
    suffix_lengths: List[int]
    num_branches: int
    total_len: int = 0  # P + sum(Si)

    # Tree cu_seqlens for the linearized layout [prefix | suffix_0 | suffix_1 | ...]
    cu_seqlens_q: Optional[torch.Tensor] = None  # [N+2]
    cu_seqlens_kv: Optional[torch.Tensor] = None  # [N+2]
    max_seqlen_q: int = 0
    max_seqlen_kv: int = 0

    # For unpacking output back to per-sample format
    original_seq_len: int = 0  # padded seq_len of original batch

    def __post_init__(self):
        self.total_len = self.prefix_len + sum(self.suffix_lengths)

    @property
    def tokens_saved(self) -> int:
        """Tokens saved across the full model (prefix computed once, not N times)."""
        return self.prefix_len * (self.num_branches - 1) if self.num_branches > 1 else 0


def detect_tree_attention(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    group_ids,
    min_prefix_ratio: float = 0.05,
) -> Optional[TreeAttentionMetadata]:
    """
    Detect if the batch forms a single tree group with shared prefix.

    Returns TreeAttentionMetadata if applicable, None otherwise.
    """
    batch_size = input_ids.shape[0]
    if batch_size <= 1:
        return None

    unique_groups = set(str(gid) for gid in group_ids)
    if len(unique_groups) != 1:
        return None

    sequences = []
    for i in range(batch_size):
        valid_len = int(attention_mask[i].sum().item())
        if valid_len == 0:
            return None
        sequences.append(input_ids[i, :valid_len])

    prefix_len = find_common_prefix_length(sequences)

    avg_len = sum(len(s) for s in sequences) / len(sequences)
    if prefix_len < min_prefix_ratio * avg_len or prefix_len < 1:
        return None

    suffix_lengths = [len(s) - prefix_len for s in sequences]
    if any(s < 0 for s in suffix_lengths):
        return None

    meta = TreeAttentionMetadata(
        prefix_len=prefix_len,
        suffix_lengths=suffix_lengths,
        num_branches=batch_size,
        original_seq_len=input_ids.shape[1],
    )

    logger.debug(
        f"Tree attention detected: {batch_size} branches, prefix_len={prefix_len}, "
        f"suffix_lens={suffix_lengths}, tokens_saved={meta.tokens_saved}"
    )
    return meta


def linearize_tree_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    meta: TreeAttentionMetadata,
    pad_factor: int = 1,
) -> Tuple[torch.Tensor, "PackedSeqParams", TreeAttentionMetadata]:
    """
    Linearize a tree group into packed format: [prefix | suffix_0 | suffix_1 | ...]

    Returns:
        packed_input_ids: [1, total_packed_len]
        packed_seq_params: PackedSeqParams for Megatron
        meta: updated TreeAttentionMetadata with cu_seqlens
    """
    from megatron.core.packed_seq_params import PackedSeqParams

    P = meta.prefix_len
    N = meta.num_branches
    device = input_ids.device

    # Build linearized token sequence
    prefix = input_ids[0, :P]
    suffix_tokens = []
    for i in range(N):
        valid_len = int(attention_mask[i].sum().item())
        suffix_tokens.append(input_ids[i, P:valid_len])

    # Pack segments with padding alignment
    cu_seqlens_list = [0]
    cu_seqlens_padded_list = [0]
    padded_segments = []

    # Segment 0: prefix
    seg_len = P
    padded_len = ((seg_len + pad_factor - 1) // pad_factor) * pad_factor
    cu_seqlens_list.append(cu_seqlens_list[-1] + seg_len)
    cu_seqlens_padded_list.append(cu_seqlens_padded_list[-1] + padded_len)
    seg = prefix
    if padded_len > seg_len:
        seg = torch.nn.functional.pad(seg, (0, padded_len - seg_len))
    padded_segments.append(seg)

    # Segments 1..N: suffixes
    for i in range(N):
        seg_len = meta.suffix_lengths[i]
        if seg_len == 0:
            cu_seqlens_list.append(cu_seqlens_list[-1])
            cu_seqlens_padded_list.append(cu_seqlens_padded_list[-1])
            continue
        padded_len = ((seg_len + pad_factor - 1) // pad_factor) * pad_factor
        cu_seqlens_list.append(cu_seqlens_list[-1] + seg_len)
        cu_seqlens_padded_list.append(cu_seqlens_padded_list[-1] + padded_len)
        seg = suffix_tokens[i]
        if padded_len > seg_len:
            seg = torch.nn.functional.pad(seg, (0, padded_len - seg_len))
        padded_segments.append(seg)

    packed_ids = torch.cat(padded_segments).unsqueeze(0)  # [1, total_packed_len]

    cu = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
    cu_pad = torch.tensor(cu_seqlens_padded_list, dtype=torch.int32, device=device)

    # Compute max seqlen from padded segments
    seg_lens_padded = (cu_pad[1:] - cu_pad[:-1]).tolist()
    max_seqlen = max(seg_lens_padded) if seg_lens_padded else 0

    # For RoPE: suffix positions are P..P+Si-1, so freqs must cover P+max(Si) positions.
    # max_seqlen_for_rope ensures Megatron generates enough rotary frequencies.
    max_suffix = max(meta.suffix_lengths) if meta.suffix_lengths else 0
    max_seqlen_for_rope = max(max_seqlen, P + max_suffix)

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu_pad,
        cu_seqlens_kv_padded=cu_pad,
        max_seqlen_q=int(max_seqlen_for_rope),
        max_seqlen_kv=int(max_seqlen_for_rope),
        qkv_format="thd",
    )
    # Attach tree metadata directly to PackedSeqParams so it flows through
    # gradient checkpointing correctly (unlike thread-local state).
    packed_seq_params._tree_meta = meta  # type: ignore[attr-defined]
    packed_seq_params._tree_cu_pad = cu_pad  # type: ignore[attr-defined]

    # Build tree cu_seqlens for flash_attn (using ACTUAL lengths, not padded)
    # Q segments: [P, S0, S1, ...]
    tree_cu_q = [0, P]
    for slen in meta.suffix_lengths:
        tree_cu_q.append(tree_cu_q[-1] + slen)
    # KV segments: [P, P+S0, P+S1, ...]
    tree_cu_kv = [0, P]
    for slen in meta.suffix_lengths:
        tree_cu_kv.append(tree_cu_kv[-1] + P + slen)

    meta.cu_seqlens_q = torch.tensor(tree_cu_q, dtype=torch.int32, device=device)
    meta.cu_seqlens_kv = torch.tensor(tree_cu_kv, dtype=torch.int32, device=device)
    meta.max_seqlen_q = max(P, max(meta.suffix_lengths) if meta.suffix_lengths else 0)
    meta.max_seqlen_kv = P + (max(meta.suffix_lengths) if meta.suffix_lengths else 0)

    return packed_ids, packed_seq_params, meta


# ---------------------------------------------------------------------------
# Custom RoPE for tree-structured positions
# ---------------------------------------------------------------------------

def _apply_tree_rotary_pos_emb(t, cu_seqlens, freqs, meta, config):
    """
    Apply RoPE with tree-correct positions for THD packed format.

    Standard per-segment RoPE gives suffix positions 0..Si-1.
    We need suffix positions P..P+Si-1 (continuing from prefix).

    Args:
        t: [total_packed_len, nh, hd]
        cu_seqlens: [num_segments+1] padded cumulative lengths
        freqs: [max_pos, 1, 1, rot_dim]
        meta: TreeAttentionMetadata
        config: TransformerConfig

    Note: cu_seqlens are PADDED lengths but freqs only covers actual positions.
    We split by padded lengths, apply RoPE only to actual tokens, keep padding unchanged.
    """
    from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

    P = meta.prefix_len
    padded_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    segments = list(torch.split(t, padded_seqlens))

    # Actual lengths: [prefix_len, suffix_0, suffix_1, ...]
    actual_seqlens = [P] + list(meta.suffix_lengths)

    rotated = []
    for i, x in enumerate(segments):
        padded_len = x.shape[0]
        if padded_len == 0:
            rotated.append(x)
            continue

        actual_len = actual_seqlens[i] if i < len(actual_seqlens) else 0
        actual_len = min(actual_len, padded_len)  # safety clamp

        if actual_len == 0:
            rotated.append(x)
            continue

        # Split into actual content and padding
        x_actual = x[:actual_len]
        x_pad = x[actual_len:]

        if i == 0:
            # Prefix segment: positions 0..P-1
            seg_freqs = freqs[:actual_len]
        else:
            # Suffix segment: positions P..P+Si-1
            seg_freqs = freqs[P : P + actual_len]

        x_rotated = _apply_rotary_pos_emb_bshd(
            x_actual.unsqueeze(1),
            seg_freqs,
            rotary_interleaved=config.rotary_interleaved,
            multi_latent_attention=config.multi_latent_attention,
        ).squeeze(1)

        if x_pad.shape[0] > 0:
            rotated.append(torch.cat([x_rotated, x_pad], dim=0))
        else:
            rotated.append(x_rotated)

    return torch.cat(rotated, dim=0)


# ---------------------------------------------------------------------------
# Core tree attention using flash_attn_varlen_func
# ---------------------------------------------------------------------------

def _tree_attention_core(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    meta: TreeAttentionMetadata,
    cu_seqlens_padded: torch.Tensor,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Core tree attention: KV expansion + flash_attn_varlen_func.

    Input Q,K,V in linearized packed layout: [prefix pad | suffix_0 pad | suffix_1 pad | ...]
    RoPE already applied with correct tree positions.

    Returns output in the same linearized packed layout.
    """
    P = meta.prefix_len
    N = meta.num_branches
    cu_pad = cu_seqlens_padded

    # Extract prefix Q,K,V (segment 0)
    prefix_q = query[:P]
    prefix_k = key[:P]
    prefix_v = value[:P]

    # Extract suffix Q,K,V from each segment
    suffix_qs, suffix_ks, suffix_vs = [], [], []
    for i in range(N):
        seg_start = cu_pad[i + 1].item()  # segment i+1 in packed layout
        slen = meta.suffix_lengths[i]
        suffix_qs.append(query[seg_start : seg_start + slen])
        suffix_ks.append(key[seg_start : seg_start + slen])
        suffix_vs.append(value[seg_start : seg_start + slen])

    # Build restructured Q: [prefix | suffix_0 | ... | suffix_{N-1}]
    new_q = torch.cat([prefix_q] + suffix_qs, dim=0)

    # Build restructured K,V with prefix duplication for each suffix
    k_segments = [prefix_k]
    v_segments = [prefix_v]
    for i in range(N):
        k_segments.append(torch.cat([prefix_k, suffix_ks[i]], dim=0))
        v_segments.append(torch.cat([prefix_v, suffix_vs[i]], dim=0))
    new_k = torch.cat(k_segments, dim=0)
    new_v = torch.cat(v_segments, dim=0)

    # Flash attention with tree cu_seqlens
    tree_out = flash_attn_varlen_func(
        new_q,
        new_k,
        new_v,
        cu_seqlens_q=meta.cu_seqlens_q,
        cu_seqlens_k=meta.cu_seqlens_kv,
        max_seqlen_q=meta.max_seqlen_q,
        max_seqlen_k=meta.max_seqlen_kv,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=True,
    )

    # Reconstruct linearized packed output
    prefix_out = tree_out[:P]
    suffix_outs = []
    offset = P
    for slen in meta.suffix_lengths:
        suffix_outs.append(tree_out[offset : offset + slen])
        offset += slen

    packed_len = cu_pad[-1].item()
    nh, hd = tree_out.shape[1], tree_out.shape[2]
    output_parts = []

    # Segment 0: prefix
    seg0_pad_len = cu_pad[1].item() - cu_pad[0].item()
    pad0 = seg0_pad_len - P
    parts = [prefix_out]
    if pad0 > 0:
        parts.append(torch.zeros(pad0, nh, hd, device=tree_out.device, dtype=tree_out.dtype))
    output_parts.append(torch.cat(parts, dim=0))

    # Segments 1..N: suffixes
    for i in range(N):
        seg_pad_len = cu_pad[i + 2].item() - cu_pad[i + 1].item()
        slen = meta.suffix_lengths[i]
        pad_len = seg_pad_len - slen
        parts = [suffix_outs[i]]
        if pad_len > 0:
            parts.append(
                torch.zeros(pad_len, nh, hd, device=tree_out.device, dtype=tree_out.dtype)
            )
        output_parts.append(torch.cat(parts, dim=0))

    packed_output = torch.cat(output_parts, dim=0)

    # Safety: ensure output matches input query length
    input_len = query.shape[0]
    if packed_output.shape[0] != input_len:
        logger.warning(
            f"Tree attention output length mismatch: got {packed_output.shape[0]}, "
            f"expected {input_len}. Padding/trimming to match."
        )
        if packed_output.shape[0] < input_len:
            pad_size = input_len - packed_output.shape[0]
            packed_output = torch.cat([
                packed_output,
                torch.zeros(pad_size, nh, hd, device=packed_output.device, dtype=packed_output.dtype),
            ], dim=0)
        else:
            packed_output = packed_output[:input_len]

    return packed_output


# ---------------------------------------------------------------------------
# Hook installation / removal
# ---------------------------------------------------------------------------

def install_tree_attention_hooks(model, meta: TreeAttentionMetadata, cu_seqlens_padded: torch.Tensor):
    """
    Install hooks on SelfAttention modules for tree attention.

    Patches both RoPE (for correct tree positions) and core_attention (for KV expansion).
    Hooks read metadata dynamically from _tree_ctx so they work correctly across
    multiple microbatches — just update _tree_ctx.meta and _tree_ctx.cu_seqlens_padded
    before each microbatch.

    Returns list of (module_name, module, original_forward) for cleanup.
    """
    import megatron.core.transformer.attention as attn_module

    hooks = []

    # Set initial context
    _tree_ctx.meta = meta
    _tree_ctx.cu_seqlens_padded = cu_seqlens_padded

    # 1. Patch apply_rotary_pos_emb at module level for tree RoPE
    _original_apply_rope = attn_module.apply_rotary_pos_emb

    def _patched_apply_rotary_pos_emb(t, freqs, config, cu_seqlens=None, **kwargs):
        # Check thread-local context (set per-microbatch before model forward)
        ctx_meta = getattr(_tree_ctx, "meta", None)
        if ctx_meta is not None and cu_seqlens is not None:
            return _apply_tree_rotary_pos_emb(t, cu_seqlens, freqs, ctx_meta, config)
        return _original_apply_rope(t, freqs, config, cu_seqlens=cu_seqlens, **kwargs)

    attn_module.apply_rotary_pos_emb = _patched_apply_rotary_pos_emb
    hooks.append(("rope_patch", attn_module, _original_apply_rope))

    # 2. Patch SelfAttention modules to set _tree_ctx from packed_seq_params
    # at the start of each forward call (before RoPE), then let
    # DotProductAttention hooks use the thread-local context.
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "SelfAttention" not in cls_name:
            continue

        original_sa_forward = module.forward

        def _make_sa_replacement(_orig_sa_forward):
            def sa_replacement(*args, **kwargs):
                # packed_seq_params flows through gradient checkpointing via closure.
                # Extract tree metadata and set thread-local context so RoPE and
                # core_attention hooks see the correct microbatch metadata.
                # IMPORTANT: always update _tree_ctx (set or clear) so that
                # non-tree microbatches don't accidentally use stale metadata.
                psp = kwargs.get("packed_seq_params", None)
                sa_meta = getattr(psp, "_tree_meta", None) if psp else None
                sa_cu_pad = getattr(psp, "_tree_cu_pad", None) if psp else None
                _tree_ctx.meta = sa_meta
                _tree_ctx.cu_seqlens_padded = sa_cu_pad
                return _orig_sa_forward(*args, **kwargs)
            return sa_replacement

        module.forward = _make_sa_replacement(original_sa_forward)
        hooks.append((name, module, original_sa_forward))

    # 3. Patch core_attention (DotProductAttention) modules for tree attention
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "DotProductAttention" not in cls_name:
            continue

        original_forward = module.forward

        def _make_replacement(_orig_forward):
            def replacement(query, key, value, *args, **kwargs):
                ctx_meta = getattr(_tree_ctx, "meta", None)
                ctx_cu_pad = getattr(_tree_ctx, "cu_seqlens_padded", None)
                if ctx_meta is not None and ctx_cu_pad is not None:
                    return _tree_attention_core(query, key, value, ctx_meta, ctx_cu_pad)
                return _orig_forward(query, key, value, *args, **kwargs)
            return replacement

        module.forward = _make_replacement(original_forward)
        hooks.append((name, module, original_forward))

    logger.debug(f"Installed tree attention hooks ({len(hooks) - 1} attention layers + RoPE patch)")
    return hooks


def set_tree_context(meta: Optional[TreeAttentionMetadata], cu_seqlens_padded: Optional[torch.Tensor] = None):
    """Update the thread-local tree attention context for the current microbatch."""
    _tree_ctx.meta = meta
    _tree_ctx.cu_seqlens_padded = cu_seqlens_padded

    # Register in the lookup table so gradient checkpointing backward can
    # recover the correct metadata even when _tree_ctx.meta has been
    # overwritten by a later microbatch.
    if not hasattr(_tree_ctx, "meta_registry"):
        _tree_ctx.meta_registry = {}
    if meta is not None and cu_seqlens_padded is not None:
        packed_len = int(cu_seqlens_padded[-1].item())
        _tree_ctx.meta_registry[packed_len] = (meta, cu_seqlens_padded)


def remove_tree_attention_hooks(hooks):
    """Restore original forward methods and unpatch RoPE."""
    import megatron.core.transformer.attention as attn_module

    _tree_ctx.meta = None
    _tree_ctx.cu_seqlens_padded = None
    _tree_ctx.meta_registry = {}

    for entry in hooks:
        name, module, orig = entry
        if name == "rope_patch":
            attn_module.apply_rotary_pos_emb = orig
        else:
            module.forward = orig


def unpack_tree_output(
    tree_output: torch.Tensor,
    meta: TreeAttentionMetadata,
    cu_seqlens_padded: torch.Tensor,
) -> torch.Tensor:
    """
    Unpack linearized tree output to per-sample batch format.

    Args:
        tree_output: [1, total_packed_len, hidden_dim] from model forward
        meta: TreeAttentionMetadata
        cu_seqlens_padded: [N+2] padded segment boundaries

    Returns:
        [N, original_seq_len, hidden_dim] — each sample has [prefix | suffix_i | padding]
    """
    hidden_dim = tree_output.shape[-1]
    N = meta.num_branches
    seq_len = meta.original_seq_len
    device = tree_output.device
    dtype = tree_output.dtype

    # Extract from linearized layout
    # tree_output is [1, packed_len, hidden_dim]
    out = tree_output.squeeze(0)  # [packed_len, hidden_dim]

    P = meta.prefix_len
    prefix_out = out[: cu_seqlens_padded[1].item()][:P]  # [P, hidden_dim]

    result = torch.zeros(N, seq_len, hidden_dim, dtype=dtype, device=device)
    for i in range(N):
        # Prefix (shared)
        result[i, :P] = prefix_out
        # Suffix
        slen = meta.suffix_lengths[i]
        if slen > 0:
            seg_start = cu_seqlens_padded[i + 1].item()
            result[i, P : P + slen] = out[seg_start : seg_start + slen]

    return result
