"""numpy reimplementation of torchao four_over_six_quantize (1x16 blocks, per-row/per-expert scalar amax)
Source: /home/scratch.hanlinb_ent_1/repos/ao-nvfp4/torchao/prototype/moe_training/nvfp4_training/four_over_six.py
  four_over_six_global_encode_scale L105-119, _fp4_rtne L122-130, _candidate_error L133-165, four_over_six_quantize L180-269,
  four_over_six_dequantize docstring L280-288 (decode = (f32(scale)*amax)*(1/(6*bound)); elem = f32(code)*decode -> bf16)
Grouped weight path: four_over_six_grouped.py _quantize_expert_weights L181-223 (1x16: whole (E,N,K) stack quantized with
  weight_amax.repeat_interleave(N) as the per-row amax), weight_amax = weight.abs().amax(dim=(1,2)) L440.
"""
import numpy as np
FP4_MAX=np.float32(6.0); E4M3_MAX=np.float32(448.0); F32_MAX=np.finfo(np.float32).max

def bf16_to_f32(u16): return (u16.astype(np.uint32)<<16).view(np.float32)
def f32_to_bf16_bits(f):
    """round-to-nearest-even f32 -> bf16 bits (finite inputs)"""
    u=f.astype(np.float32).view(np.uint32).astype(np.uint64)
    lsb=(u>>16)&1
    u=(u+0x7FFF+lsb)>>16
    return u.astype(np.uint16)

def e4m3_rne(x):
    """f32 (0<=x<=448) -> nearest float8_e4m3fn value (as f32), RNE. bias 7, 3 mantissa bits, min normal 2^-6, subnormal step 2^-9."""
    x=x.astype(np.float32)
    m,e=np.frexp(x)            # x = m*2^e, m in [0.5,1)
    e=e-1                       # x = (2m)*2^e, 2m in [1,2)
    step=np.where(e< -6, np.float32(2.0**-9), np.exp2(e-3).astype(np.float32))
    q=np.rint(x/step)*step
    return np.minimum(q, E4M3_MAX).astype(np.float32)

def e2m1_rne(y):
    """f32 -> nearest E2M1 value (as f32) with round-to-nearest-even on the code (cvt.rn.satfinite.e2m1x2.f32)."""
    a=np.abs(np.clip(y,-6,6)).astype(np.float32)
    # map |y| to a continuous 'code coordinate' whose integer points are the E2M1 codes 0..7
    c=np.where(a<2, a/0.5, np.where(a<4, 4+(a-2), 6+(a-4)/2))
    ci=np.rint(c)
    vals=np.array([0,0.5,1,1.5,2,3,4,6],dtype=np.float32)
    v=vals[ci.astype(np.int64)]
    return np.copysign(v, y).astype(np.float32), ci.astype(np.int8)*np.where(y<0,-1,1).astype(np.int8)

def fo6_quantize_rows(x, row_amax, err_mode='mse', bound=256):
    """x: (R, C) f32, row_amax: (R,) f32. returns codes (R,C) int8 signed-code, scales_e4m3 (R,C//16) f32, deq (R,C) f32 (before bf16 cast)"""
    R,C=x.shape; xf=x.astype(np.float32).reshape(R,C//16,16)
    amax=row_amax.astype(np.float32)
    s_enc=np.float32(bound*6.0)/amax
    s_enc=np.where((amax==0)|(s_enc==0), np.float32(1.0), np.minimum(s_enc,F32_MAX)).astype(np.float32)   # L114-119
    s_enc=s_enc.reshape(R,1); err_amax=amax.reshape(R,1)
    block_amax=np.abs(xf).max(axis=-1)                                  # L238
    base=(block_amax/FP4_MAX)*s_enc                                     # L245
    scale6=e4m3_rne(np.minimum(base,E4M3_MAX))                          # L246
    scale4=e4m3_rne(np.minimum(base*np.float32(1.5),E4M3_MAX))          # L247
    s_dec=np.float32(1.0)/s_enc                                         # L248
    with np.errstate(divide='ignore'):
        inv6=np.minimum(np.float32(1.0)/(scale6*s_dec),F32_MAX).astype(np.float32)   # L249
        inv4=np.minimum(np.float32(1.0)/(scale4*s_dec),F32_MAX).astype(np.float32)
    v6,c6=e2m1_rne(xf*inv6[...,None]); v4,c4=e2m1_rne(xf*inv4[...,None])          # L252-253
    denom=np.float32(6.0*bound)
    def err(vals,sf):
        e=np.zeros(vals.shape[:2],dtype=np.float32)
        for i in range(16):                                              # L157-165 sequential fp32 adds
            d=((vals[...,i]*sf)*err_amax)/denom - xf[...,i]
            e=e+(d*d if err_mode=='mse' else np.abs(d))
        return e
    pick4=err(v4,scale4)<err(v6,scale6)                                 # L265
    codes=np.where(pick4[...,None],c4,c6); vals=np.where(pick4[...,None],v4,v6); scales=np.where(pick4,scale4,scale6)
    decode=(scales*err_amax)*(np.float32(1.0)/denom)                    # dequantize docstring L284-286
    deq=(vals*decode[...,None]).reshape(R,C)
    return codes.reshape(R,C), scales, deq, pick4

def quantize_expert_stack(w):
    """w: (E,N,K) f32 -> per-expert amax over (N,K) (grouped L440), 1x16 rows (grouped _quantize_expert_weights L197-206)"""
    E,N,K=w.shape
    amax=np.abs(w).reshape(E,-1).max(axis=1)
    codes,scales,deq,pick4=fo6_quantize_rows(w.reshape(E*N,K), np.repeat(amax,N))
    return codes.reshape(E,N,K), scales.reshape(E,N,K//16), deq.reshape(E,N,K), pick4.reshape(E,N,K//16), amax
