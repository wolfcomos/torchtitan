# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Opt-in fused MXFP8 MLA Q-projection + RoPE + quantize (cuDNN-frontend).

Activate with::

    --override.imports torchtitan.overrides.mxfp8_mla_q_rope.mxfp8_mla_q_rope

Replaces the DeepSeek-V3 ``Attention`` Q path (``wq`` GEMM -> head view ->
ComplexRoPE on the rotary tail) with one torchao custom op wrapping the
cudnn-frontend ``gemm_proj_rope_mxfp8_wrapper_sm100`` kernel (the fusion
TransformerEngine PR #3303 wires for DeepSeek-V3 MLA MXFP8 training): MXFP8
projection GEMM + per-head YARN RoPE + rowwise AND columnwise MXFP8
quantization of Q, with no BF16 Q round trip inside the kernel.

Because torchtitan's inner attention runs BF16 (no MXFP8 SDPA backend), the
rowwise MXFP8 Q is dequantized back to BF16 for ``inner_attention``; the
columnwise output (the FP8-attention-backward operand) is currently unused.
The extra quantize->dequantize round trip on Q is the recipe's intended
semantic (an MXFP8 attention consumer would read the FP8 codes directly) and
is a real numerics delta versus the stock MXFP8Linear path.

RoPE convention: the kernel reads the rotary tail INTERLEAVED (torchtitan's
ComplexRoPE adjacent-pair convention) but writes it HALF-CONCATENATED
(Megatron YARN layout). Q therefore leaves this module with a permuted rotary
tail, and the K rotary tail is permuted identically (a cheap gather on the
shared single-head ``k_pe``), which leaves every QK^T dot product exactly
invariant. The backward inverse-rotates the half-concatenated Q-tail gradient
back to the interleaved layout before the MXFP8 dgrad/wgrad GEMMs, which
reproduce ``torchao.prototype.moe_training.mxfp8_linear``'s backward verbatim
(STE through the output quantize, the TE #3330 ``bf16_backward`` convention).

The override keeps the stock Attention parameters and state-dict layout; it
reads ``self.wq.weight`` directly and never calls ``wq.forward``. Fail-loud:
every configuration the kernel cannot execute raises at config-application
time (or at first forward for runtime-only contracts); there is no silent
fallback.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch.distributed.tensor import DTensor

# Importing the wrapper module registers the torchao:: custom op; the cudnn
# package is only imported lazily inside the op body at first launch. The
# module is newer than every torchao release, so its absence must surface as
# the factory's actionable config-time error (with this reason), not an
# ImportError at override-import time.
try:
    from torchao.prototype.moe_training.kernels.mxfp8.cudnn_mla_q_proj_rope import (
        is_supported,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        TOKEN_ALIGNMENT,
    )
except ImportError as _exc:
    is_supported = None
    _TORCHAO_MLA_Q_ROPE_OPS_AVAILABLE = False
    _TORCHAO_MLA_Q_ROPE_UNAVAILABLE_REASON = (
        "the installed torchao has no torchao.prototype.moe_training.kernels."
        f"mxfp8.cudnn_mla_q_proj_rope module (a torchao build that ships the "
        f"fused MLA Q-projection custom op is required): {_exc}"
    )
else:
    _TORCHAO_MLA_Q_ROPE_OPS_AVAILABLE = True
    _TORCHAO_MLA_Q_ROPE_UNAVAILABLE_REASON = ""
import torchao.prototype.mx_formats.kernels  # noqa: F401  registers triton_mxfp8_dequant_dim0
from torchao.prototype.mx_formats.config import (
    MXFP8Dim0CastKernelChoice,
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

from torchtitan.config import derive, override
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.deepseek_v3.model import Attention

__all__ = [
    "MXFP8MLAQRopeAttention",
    "mxfp8_mla_q_rope",
]

_E4M3 = torch.float8_e4m3fn
_BLOCK = 32


def _rope_tables(cache_c: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Kernel cos/sin tables + fp32 halves from a per-token complex cache.

    ``cache_c`` is the ``(tokens, 1, rope_dim / 2)`` complex64 slice stock
    ``ComplexRoPE._reshape_cache`` produces. Returns ``(cos, sin, cos32,
    sin32)``: BF16 ``[tokens, rope_dim]`` duplicated-freq halves tables for
    the kernel (``cos[:, :32] == cos[:, 32:]`` -- its rotation identity
    requires the duplication) and the fp32 ``[tokens, rope_dim / 2]`` halves
    the backward inverse rotation uses.
    """
    cache_r = torch.view_as_real(cache_c.squeeze(1))
    cos32 = cache_r[..., 0]
    sin32 = cache_r[..., 1]
    cos = torch.cat([cos32, cos32], dim=-1).bfloat16().contiguous()
    sin = torch.cat([sin32, sin32], dim=-1).bfloat16().contiguous()
    return cos, sin, cos32, sin32


class _MXFP8MLAQProjRope(torch.autograd.Function):
    """wq GEMM + RoPE + dual MXFP8 quantize forward; MXFP8Linear-class backward.

    Forward quantizes x and w rowwise exactly as ``mxfp8_linear.mx_mm`` does,
    launches the fused kernel, and dequantizes the rowwise Q back to BF16 for
    the BF16 inner attention (STE through that round trip). Backward
    inverse-rotates the half-concatenated rotary-tail gradient back to the
    interleaved pre-RoPE layout in fp32, then reproduces ``mx_mm.backward``'s
    dgrad/wgrad composition (dim0 TRITON / dim1 CUDA casts, RCEIL) verbatim.
    """

    @staticmethod
    def forward(ctx, x, w, cos, sin, cos32, sin32):
        x_mx = MXTensor.to_mx(
            x,
            _E4M3,
            _BLOCK,
            ScaleCalculationMode.RCEIL,
            KernelPreference.AUTO,
            mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
        )
        w_mx = MXTensor.to_mx(
            w,
            _E4M3,
            _BLOCK,
            ScaleCalculationMode.RCEIL,
            KernelPreference.AUTO,
            mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
        )
        q_row_q, q_row_sf, _q_col_q, _q_col_sf = (
            torch.ops.torchao.mxfp8_mla_q_proj_rope_cudnn(
                x_mx.qdata,
                w_mx.qdata,
                cos,
                sin,
                x_mx.scale.view(torch.uint8),
                w_mx.scale.view(torch.uint8),
            )
        )
        # The columnwise outputs are the FP8-attention-backward operands;
        # torchtitan has no MXFP8 SDPA consumer, so they are dropped here.
        tokens, num_heads = q_row_q.shape[0], q_row_q.shape[1]
        q = torch.ops.torchao.triton_mxfp8_dequant_dim0(
            q_row_q,
            q_row_sf.view(tokens * num_heads, q_row_sf.shape[-1]),
            torch.bfloat16,
            _BLOCK,
        )
        ctx.save_for_backward(x, w, cos32, sin32)
        return q

    @staticmethod
    def backward(ctx, grad_q):
        x, w, cos32, sin32 = ctx.saved_tensors
        g = grad_q.contiguous()

        # Inverse-rotate the half-concatenated rotary-tail gradient back to
        # the interleaved pre-RoPE layout: for y1 = x1*c - x2*s and
        # y2 = x2*c + x1*s, dx1 = g1*c + g2*s and dx2 = g2*c - g1*s.
        half = QK_ROPE_HEAD_DIM // 2
        g1 = g[..., QK_NOPE_HEAD_DIM : QK_NOPE_HEAD_DIM + half].float()
        g2 = g[..., QK_NOPE_HEAD_DIM + half :].float()
        c = cos32.unsqueeze(1)
        s = sin32.unsqueeze(1)
        dx1 = g1 * c + g2 * s
        dx2 = g2 * c - g1 * s
        tail = torch.stack([dx1, dx2], dim=-1).flatten(-2).to(g.dtype)
        dy = (
            torch.cat([g[..., :QK_NOPE_HEAD_DIM], tail], dim=-1)
            .reshape(g.shape[0], -1)
            .contiguous()
        )

        # From here this is mxfp8_linear.mx_mm.backward verbatim.
        go_dim0 = MXTensor.to_mx(
            dy,
            _E4M3,
            _BLOCK,
            ScaleCalculationMode.RCEIL,
            KernelPreference.AUTO,
            mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
        )
        w_dim1 = _to_mxfp8_dim1_kernel_wrapper(
            w,
            _BLOCK,
            _E4M3,
            w.dtype,
            KernelPreference.AUTO,
            MXFP8Dim1CastKernelChoice.CUDA,
            ScaleCalculationMode.RCEIL,
        )
        dx = torch.mm(go_dim0, w_dim1.t())

        go_dim1 = _to_mxfp8_dim1_kernel_wrapper(
            dy,
            _BLOCK,
            _E4M3,
            dy.dtype,
            KernelPreference.AUTO,
            MXFP8Dim1CastKernelChoice.CUDA,
            ScaleCalculationMode.RCEIL,
        )
        x_t_dim0 = _to_mxfp8_dim1_kernel_wrapper(
            x,
            _BLOCK,
            _E4M3,
            x.dtype,
            KernelPreference.AUTO,
            MXFP8Dim1CastKernelChoice.CUDA,
            ScaleCalculationMode.RCEIL,
        ).t()
        dw = torch.mm(go_dim1, x_t_dim0)

        return dx, dw, None, None, None, None


class MXFP8MLAQRopeAttention(Attention):
    """Stock DeepSeek-V3 attention with the fused MXFP8 Q-projection path."""

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        if not isinstance(self.rope, ComplexRoPE):
            raise TypeError(
                "MXFP8MLAQRopeAttention requires ComplexRoPE, got "
                f"{type(self.rope).__name__}."
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        if num_tokens % TOKEN_ALIGNMENT != 0:
            raise RuntimeError(
                "MXFP8MLAQRopeAttention requires the flattened token count to "
                f"be a multiple of {TOKEN_ALIGNMENT} (the fused kernel's "
                f"TILE_M; no tail handling), got {num_tokens}. Adjust "
                "training.num_tokens_per_microbatch_per_dp_rank or remove the "
                "override."
            )
        w = self.wq.weight
        if isinstance(w, DTensor):
            raise RuntimeError(
                "MXFP8MLAQRopeAttention does not support a tensor-parallel "
                "wq (the quantize kernels and the fused op consume plain "
                "local tensors); run with tensor_parallel_degree=1 or remove "
                "the override."
            )

        # Per-token complex rope cache through the stock machinery (positions
        # wrapping, None handling, and bounds check included). The empty
        # tensor only supplies the (tokens, 1, rope_dim) shape.
        cache_c = self.rope._reshape_cache(
            x.new_empty(num_tokens, 1, self.qk_rope_head_dim), positions
        )
        cos, sin, cos32, sin32 = _rope_tables(cache_c)

        q = _MXFP8MLAQProjRope.apply(x, w, cos, sin, cos32, sin32)
        with spmd.local():
            if spmd.is_type_checking():
                spmd.assert_type(
                    q,
                    spmd.V,
                    spmd.PartitionSpec(("dp", "cp"), "tp", None),
                )

        # Key-value projection (stock, except the rotated k_pe tail is
        # permuted to the kernel's half-concatenated layout so QK^T stays
        # exactly invariant).
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pe = k_pe.unsqueeze(1)
        k_pe = ComplexRoPE.apply_rotary_emb(k_pe, k_pe, cache_c)[0]
        k_pe = torch.cat([k_pe[..., 0::2], k_pe[..., 1::2]], dim=-1)

        kv = self.wkv_b(self.kv_norm(kv))

        with (
            spmd.local()
        ):  # QKV even shard unflatten, but the expand is truly local SPMD
            kv = kv.view(num_tokens, -1, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1)], dim=-1)
            if spmd.is_type_checking() and not torch.compiler.is_compiling():
                for t in [k, v]:
                    spmd.assert_type(
                        t,
                        spmd.V,
                        spmd.PartitionSpec(("dp", "cp"), "tp", None),
                    )

        output = self.inner_attention(
            q, k, v, attention_masks=attention_masks, scale=self.softmax_scale
        ).contiguous()
        output = output.view(num_tokens, -1)
        return self.wo(output)


@override(
    target=Attention.Config,
    description="Fused MXFP8 MLA Q-projection + RoPE + quantize (cuDNN-frontend).",
)
def mxfp8_mla_q_rope(cfg: Attention.Config) -> MXFP8MLAQRopeAttention.Config:
    if not (
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)
    ):
        raise ValueError(
            "mxfp8_mla_q_rope requires CUDA device capability exactly (10, 0) "
            "(the torchao op wraps the cudnn gemm_proj_rope_mxfp8_wrapper_sm100 "
            "kernel); remove the override or run on supported hardware."
        )
    if not _TORCHAO_MLA_Q_ROPE_OPS_AVAILABLE:
        raise ValueError(
            f"mxfp8_mla_q_rope: {_TORCHAO_MLA_Q_ROPE_UNAVAILABLE_REASON}"
        )
    if type(cfg) is not Attention.Config:
        raise ValueError(
            "mxfp8_mla_q_rope targets the stock DeepSeek-V3 Attention.Config, "
            f"got {type(cfg).__qualname__}; narrow this override's fqns or "
            "remove the conflicting override."
        )
    if not isinstance(cfg.rope, ComplexRoPE.Config):
        raise ValueError(
            "mxfp8_mla_q_rope requires ComplexRoPE (the kernel implements its "
            f"adjacent-pair rotation), got {type(cfg.rope).__qualname__}."
        )
    if cfg.q_lora_rank != 0:
        raise ValueError(
            "mxfp8_mla_q_rope currently fuses only the direct wq projection "
            f"(q_lora_rank == 0); got q_lora_rank={cfg.q_lora_rank}, whose Q "
            "path is wq_b(q_norm(wq_a(x))). Wire the wq_b seam before using "
            "this override on 236B/671B-class configs."
        )
    if not is_supported(
        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.dim, cfg.n_heads
    ):
        raise ValueError(
            "mxfp8_mla_q_rope: unsupported dims for the fused kernel "
            f"(qk_nope_head_dim={cfg.qk_nope_head_dim}, qk_rope_head_dim="
            f"{cfg.qk_rope_head_dim}, dim={cfg.dim}, n_heads={cfg.n_heads}); "
            "the kernel requires the fixed 128+64 head geometry, dim % 128 "
            "== 0, and an even head count."
        )
    return derive(cfg, MXFP8MLAQRopeAttention.Config)
