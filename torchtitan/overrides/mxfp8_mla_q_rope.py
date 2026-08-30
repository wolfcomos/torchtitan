# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Opt-in fused MXFP8 MLA Q-projection + RoPE + quantize (cuDNN-frontend).

Activate with::

    --override.imports torchtitan.overrides.mxfp8_mla_q_rope.mxfp8_mla_q_rope

Replaces the DeepSeek-V3 ``Attention`` Q projection (``wq`` GEMM when
``q_lora_rank == 0``; the ``wq_b`` GEMM after the stock ``wq_a``/``q_norm``
low-rank stage on 236B/671B-class configs) -> head view -> ComplexRoPE on
the rotary tail with one torchao custom op wrapping the cudnn-frontend
``gemm_proj_rope_mxfp8_wrapper_sm100`` kernel (the fusion TransformerEngine
PR #3303 wires for DeepSeek-V3 MLA MXFP8 training): MXFP8 projection GEMM +
per-head YARN RoPE + rowwise AND columnwise MXFP8 quantization of Q, with no
BF16 Q round trip inside the kernel.

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
reads the Q projection weight (``wq.weight``, or ``wq_b.weight`` on the
low-rank path) directly and never calls that linear's forward. Fail-loud:
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
# the factory's actionable config-time error, not an ImportError at
# override-import time.
try:
    from torchao.prototype.moe_training.kernels.mxfp8.cudnn_mla_q_proj_rope import (
        is_supported,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        SCALE_BLOCK_SIZE as _BLOCK,
        TOKEN_ALIGNMENT,
    )
except ImportError as _exc:
    _TORCHAO_IMPORT_ERROR = _exc
else:
    _TORCHAO_IMPORT_ERROR = None
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
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.deepseek_v3.model import Attention

__all__ = [
    "MLAQRopeSelectiveAC",
    "MXFP8MLAQRopeAttention",
    "MXFP8MLAQRopeTEAttention",
    "mxfp8_mla_q_rope",
    "mxfp8_mla_q_rope_te_attn",
]

_E4M3 = torch.float8_e4m3fn


def _to_mx_dim0(t: torch.Tensor) -> MXTensor:
    return MXTensor.to_mx(
        t,
        _E4M3,
        _BLOCK,
        ScaleCalculationMode.RCEIL,
        KernelPreference.AUTO,
        mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
    )


def _to_mx_dim1(t: torch.Tensor) -> MXTensor:
    return _to_mxfp8_dim1_kernel_wrapper(
        t,
        _BLOCK,
        _E4M3,
        t.dtype,
        KernelPreference.AUTO,
        MXFP8Dim1CastKernelChoice.CUDA,
        ScaleCalculationMode.RCEIL,
    )


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
    """wq GEMM + RoPE + dual MXFP8 quantize forward; MXFP8Linear-class
    backward (dim0 TRITON / dim1 CUDA casts, RCEIL)."""

    @staticmethod
    def forward(ctx, x, w, cos, sin, cos32, sin32):
        x_mx = _to_mx_dim0(x)
        w_mx = _to_mx_dim0(w)
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
        go_dim0 = _to_mx_dim0(dy)
        w_dim1 = _to_mx_dim1(w)
        dx = torch.mm(go_dim0, w_dim1.t())

        go_dim1 = _to_mx_dim1(dy)
        x_t_dim0 = _to_mx_dim1(x).t()
        dw = torch.mm(go_dim1, x_t_dim0)

        return dx, dw, None, None, None, None


class _MXFP8MLAQProjRopeCodes(torch.autograd.Function):
    """Arm-E variant of :class:`_MXFP8MLAQProjRope`: same forward op and same
    backward composition, but returns the RAW fp8 codes/scales (for a direct
    MXFP8-attention consumer) plus an uninitialized BF16 host tensor that
    exists only as the autograd edge -- its values are never read; TE's
    MXFP8 attention consumes the codes, and dQ is routed back through the
    host into the inverse-rope + MXFP8 dgrad/wgrad backward. This removes
    the dequant bridge entirely."""

    @staticmethod
    def forward(ctx, x, w, cos, sin, cos32, sin32):
        x_mx = _to_mx_dim0(x)
        w_mx = _to_mx_dim0(w)
        q_row_q, q_row_sf, q_col_q, q_col_sf = (
            torch.ops.torchao.mxfp8_mla_q_proj_rope_cudnn(
                x_mx.qdata,
                w_mx.qdata,
                cos,
                sin,
                x_mx.scale.view(torch.uint8),
                w_mx.scale.view(torch.uint8),
            )
        )
        host = q_row_q.new_empty(q_row_q.shape, dtype=torch.bfloat16)
        ctx.save_for_backward(x, w, cos32, sin32)
        ctx.mark_non_differentiable(q_row_q, q_row_sf, q_col_q, q_col_sf)
        return host, q_row_q, q_row_sf, q_col_q, q_col_sf

    @staticmethod
    def backward(ctx, grad_host, *_unused):
        dx, dw, *rest = _MXFP8MLAQProjRope.backward(ctx, grad_host)
        return dx, dw, None, None, None, None


class _STEToMX(torch.autograd.Function):
    """Straight-through bridge: forward a pre-built MXFP8Tensor into TE's
    attention; route its gradient back to the BF16 host (reshaped to the
    host's layout). The TE#3330 STE convention, applied at the attention
    boundary."""

    @staticmethod
    def forward(ctx, host, mx_tensor):
        ctx.host_shape = host.shape
        return mx_tensor

    @staticmethod
    def backward(ctx, grad):
        return grad.reshape(ctx.host_shape), None


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
        if self.q_lora_rank == 0:
            q_in, w = x, self.wq.weight
        else:
            # 236B/671B-class Q path: the fused op replaces the wq_b GEMM;
            # wq_a and q_norm stay stock (wq_a keeps whatever converter
            # quantization the config applied to it).
            q_in = self.q_norm(self.wq_a(x))
            w = self.wq_b.weight
        if isinstance(w, DTensor):
            raise RuntimeError(
                "MXFP8MLAQRopeAttention does not support a tensor-parallel "
                "Q projection (the quantize kernels and the fused op consume "
                "plain local tensors); run with tensor_parallel_degree=1 or "
                "remove the override."
            )

        # Per-token complex rope cache through the stock machinery (positions
        # wrapping, None handling, and bounds check included). The empty
        # tensor only supplies the (tokens, 1, rope_dim) shape.
        cache_c = self.rope._reshape_cache(
            x.new_empty(num_tokens, 1, self.qk_rope_head_dim), positions
        )
        cos, sin, cos32, sin32 = _rope_tables(cache_c)

        q = _MXFP8MLAQProjRope.apply(q_in, w, cos, sin, cos32, sin32)
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


class MXFP8MLAQRopeTEAttention(MXFP8MLAQRopeAttention):
    """Arm E: the fused Q projection feeds TE's MXFP8 fused attention
    directly (TE#2719) -- no dequant bridge, and the columnwise Q output
    becomes the backward's dK operand instead of dead weight.

    The flat token-major batch is viewed as ``bshd`` with
    ``s = te_attn_seq_len`` (the recipe's max_context_length), attention is
    plain causal PER WINDOW: packed documents inside one window may attend
    across document boundaries, unlike the flex block mask. That is a
    disclosed semantics delta of this arm (it also does MORE attention work
    than the doc-masked baseline, never less). K/V are quantized at the
    attention boundary (``mxfp8_quantize_only`` + one batched
    ``mxfp8_transpose_swizzle`` with Q's wrapped codes); backward runs with
    ``bf16_backward=True`` so dQ/dK/dV arrive in BF16 for the existing
    inverse-rope / wkv autograd chain (STE through the attention-input
    quantizes)."""

    @dataclass(kw_only=True, slots=True)
    class Config(MXFP8MLAQRopeAttention.Config):
        # Per-sequence window the flat batch is folded into (bshd); the
        # flavor must set this to the recipe's max_context_length.
        te_attn_seq_len: int = 4096

    def __init__(self, config: Config):
        super().__init__(config)
        self.te_attn_seq_len = config.te_attn_seq_len
        try:
            import transformer_engine.pytorch as te
            import transformer_engine_torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "MXFP8MLAQRopeTEAttention requires transformer_engine with "
                "MXFP8 attention support (TE#2719, TE >= 2.2 dev with "
                "cuDNN 9.21+); install it or use the non-TE override."
            ) from exc
        self._te = te
        # Stored OUTSIDE the nn.Module registry (plain __dict__): TE's
        # DotProductAttention is not a torchtitan-protocol Module and holds
        # no parameters/buffers we need in the state dict, sharding walk, or
        # trainer validation.
        self.__dict__["te_dpa"] = te.DotProductAttention(
            num_attention_heads=config.n_heads,
            kv_channels=(
                config.qk_nope_head_dim + config.qk_rope_head_dim,
                config.v_head_dim,
            ),
            attention_dropout=0.0,
            qkv_format="bshd",
            softmax_scale=self.softmax_scale,
        )

    def _mxfp8_attention_inputs(self, q_parts, k, v, b, s):
        """Wrap the fused op's Q codes + freshly quantized K/V into
        attention-ready MXFP8Tensors (scale_inv permuted to BHSD, padded,
        GEMM-swizzled -- the ``combine_and_quantize`` contract)."""
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.attention.dot_product_attention.utils import (
            mxfp8_quantize_only,
            mxfp8_transpose_swizzle,
        )
        from transformer_engine.pytorch.tensor.mxfp8_tensor import (
            MXFP8Quantizer,
            MXFP8Tensor,
        )

        q_row, q_row_sf, q_col, q_col_sf = q_parts
        nh = q_row.shape[1]
        d = q_row.shape[2]
        q_mx = MXFP8Tensor(
            shape=(b, s, nh, d),
            dtype=torch.bfloat16,
            rowwise_data=q_row.view(torch.uint8).view(b, s, nh, d),
            rowwise_scale_inv=q_row_sf.view(b, s, nh, d // _BLOCK),
            columnwise_data=q_col.view(torch.uint8).view(b, s, nh, d),
            columnwise_scale_inv=q_col_sf.view(b, s // _BLOCK, nh, d),
            quantizer=MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True
            ),
            requires_grad=False,
            fp8_dtype=tex.DType.kFloat8E4M3,
            with_gemm_swizzled_scales=False,
        )
        quantizers = [
            MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True
            )
            for _ in range(2)
        ]
        k_mx, v_mx = mxfp8_quantize_only(
            [(k.detach(), quantizers[0]), (v.detach(), quantizers[1])], "bshd"
        )
        mxfp8_transpose_swizzle([q_mx, k_mx, v_mx], "bshd")
        return q_mx, k_mx, v_mx

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        s = self.te_attn_seq_len
        if num_tokens % max(s, TOKEN_ALIGNMENT) != 0 or s % _BLOCK != 0:
            raise RuntimeError(
                "MXFP8MLAQRopeTEAttention requires the flattened token count "
                f"to fold into [b, te_attn_seq_len={s}] windows (and s % "
                f"{_BLOCK} == 0); got {num_tokens} tokens. Set "
                "te_attn_seq_len to the recipe's max_context_length."
            )
        b = num_tokens // s

        if self.q_lora_rank == 0:
            q_in, w = x, self.wq.weight
        else:
            q_in = self.q_norm(self.wq_a(x))
            w = self.wq_b.weight
        if isinstance(w, DTensor):
            raise RuntimeError(
                "MXFP8MLAQRopeTEAttention does not support a tensor-parallel "
                "Q projection; run with tensor_parallel_degree=1."
            )

        cache_c = self.rope._reshape_cache(
            x.new_empty(num_tokens, 1, self.qk_rope_head_dim), positions
        )
        cos, sin, cos32, sin32 = _rope_tables(cache_c)

        q_host, *q_parts = _MXFP8MLAQProjRopeCodes.apply(
            q_in, w, cos, sin, cos32, sin32
        )

        # KV path: stock, with the k_pe halves permute (QK^T invariant).
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.unsqueeze(1)
        k_pe = ComplexRoPE.apply_rotary_emb(k_pe, k_pe, cache_c)[0]
        k_pe = torch.cat([k_pe[..., 0::2], k_pe[..., 1::2]], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(num_tokens, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1)], dim=-1)

        k4 = k.view(b, s, k.shape[1], k.shape[2]).contiguous()
        v4 = v.reshape(b, s, v.shape[1], v.shape[2]).contiguous()
        with torch.no_grad():
            q_mx, k_mx, v_mx = self._mxfp8_attention_inputs(q_parts, k4, v4, b, s)

        qb = _STEToMX.apply(q_host, q_mx)
        kb = _STEToMX.apply(k4, k_mx)
        vb = _STEToMX.apply(v4, v_mx)
        with self._te.fp8_autocast(
            enabled=True, fp8_recipe=self._te_recipe()
        ):
            out = self.te_dpa(qb, kb, vb, qkv_format="bshd", bf16_backward=True)
        output = out.reshape(num_tokens, -1)
        return self.wo(output)

    def _te_recipe(self):
        from transformer_engine.common.recipe import MXFP8BlockScaling

        return MXFP8BlockScaling(fp8_dpa=True)


class MLAQRopeSelectiveAC(SelectiveAC):
    """SelectiveAC that additionally saves the fused Q-projection composite.

    The fused op and the dequant bridge are custom ops outside the stock SAC
    save set, so stock ``SelectiveAC`` recomputes the whole Q composite in
    backward -- two fused-kernel launches (and two wrapper CPU enqueues) per
    layer-step. Saving both ops' outputs trades memory (the op's fp8
    codes/scales, including the currently-unused columnwise pair, plus the
    BF16 dequantized Q) for that recompute. The stock every-second-matmul
    policy is untouched.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(SelectiveAC.Config):
        pass

    def get_save_ops(self) -> set:
        save_ops = set(super().get_save_ops())
        save_ops.add(torch.ops.torchao.mxfp8_mla_q_proj_rope_cudnn.default)
        save_ops.add(torch.ops.torchao.triton_mxfp8_dequant_dim0.default)
        return save_ops


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
    if _TORCHAO_IMPORT_ERROR is not None:
        raise ValueError(
            "mxfp8_mla_q_rope: the installed torchao has no torchao.prototype."
            "moe_training.kernels.mxfp8.cudnn_mla_q_proj_rope module (a "
            "torchao build that ships the fused MLA Q-projection custom op is "
            "required)."
        ) from _TORCHAO_IMPORT_ERROR
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
    # The fused GEMM's contraction dim is the Q projection's input width:
    # dim for the direct wq path, q_lora_rank for the wq_b(q_norm(wq_a(x)))
    # low-rank path (236B/671B-class configs).
    in_features = cfg.q_lora_rank if cfg.q_lora_rank != 0 else cfg.dim
    if not is_supported(
        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, in_features, cfg.n_heads
    ):
        raise ValueError(
            "mxfp8_mla_q_rope: unsupported dims for the fused kernel "
            f"(qk_nope_head_dim={cfg.qk_nope_head_dim}, qk_rope_head_dim="
            f"{cfg.qk_rope_head_dim}, in_features={in_features}, "
            f"n_heads={cfg.n_heads}); the kernel requires the fixed 128+64 "
            "head geometry, in_features % 128 == 0 (dim when q_lora_rank == "
            "0, else q_lora_rank), and an even head count of at least 8 "
            "(smaller counts produce numerically corrupt kernel output)."
        )
    return derive(cfg, MXFP8MLAQRopeAttention.Config)


@override(
    target=Attention.Config,
    description=(
        "Fused MXFP8 MLA Q-projection + RoPE feeding TE MXFP8 fused "
        "attention directly (no dequant bridge; causal-per-window)."
    ),
)
def mxfp8_mla_q_rope_te_attn(cfg: Attention.Config) -> MXFP8MLAQRopeTEAttention.Config:
    base = mxfp8_mla_q_rope(cfg)
    try:
        import transformer_engine.pytorch  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "mxfp8_mla_q_rope_te_attn requires transformer_engine with MXFP8 "
            "attention (TE#2719); install it or use mxfp8_mla_q_rope."
        ) from exc
    return derive(base, MXFP8MLAQRopeTEAttention.Config)
