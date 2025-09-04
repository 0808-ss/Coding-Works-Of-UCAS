# """
# Fused Attention
# ===============
 
# This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
# Credits: OpenAI kernel team
 
# Extra Credits:
# - Original flash attention paper (https://arxiv.org/abs/2205.14135)
# - Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
 
# """
 
# import torch
# import torch_mlu
 
# import triton
# import triton.language as tl
# #from genesis.Python.Test.Common.utils import reset_tmp_dir
# import time
# import numpy as np
 
# @triton.jit
# def _attn_fwd_inner(
#         acc,
#         l_i,
#         m_i,
#         q,  #
#         K_block_ptr,
#         V_block_ptr,  #
#         start_m,
#         qk_scale,  #
#         BLOCK_M: tl.constexpr,
#         BLOCK_DMODEL: tl.constexpr,
#         BLOCK_N: tl.constexpr,  #
#         STAGE: tl.constexpr,
#         offs_m: tl.constexpr,
#         offs_n: tl.constexpr,  #
#         N_CTX: tl.constexpr,
#         IS_DIVISIBLE: tl.constexpr):
#     # range of values handled by this stage
#     if STAGE == 1:
#         lo, hi = 0, start_m * BLOCK_M
#     elif STAGE == 2:
#         lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
#         lo = tl.multiple_of(lo, BLOCK_M)
#     # causal = False
#     else:
#         lo, hi = 0, N_CTX
#     K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
#     V_block_ptr = tl.advance(V_block_ptr, (0, lo))
#     # loop over k, v and update accumulator
#     for start_n in range(lo, hi, BLOCK_N):         # 处理 kv上的 (N_CTX/BLOCK_N) 数据
#         #tl.device_print("------------>\n")
#         start_n = tl.multiple_of(start_n, BLOCK_N)
#         # -- compute qk ----
#         if IS_DIVISIBLE:
#             k = tl.load(K_block_ptr)
#         else:
#             k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
#         qk = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
#         qk += tl.dot(k, q)
#         if STAGE == 2:#用掩码阻止某些计算
#             mask = offs_m[None, :] >= (start_n + offs_n[:, None])
#             qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
#             m_ij = tl.maximum(m_i, tl.max(qk, 0))
#             qk -= m_ij[None, :]
#         else:
#             m_ij = tl.maximum(m_i, tl.max(qk * qk_scale, 0))
#             qk = qk * qk_scale - m_ij[None, :]
#         p = tl.exp2(qk)
#         l_ij = tl.sum(p, 0)
#         # -- update m_i and l_i
#         alpha = tl.exp2(m_i - m_ij)
#         if IS_DIVISIBLE:
#             v = tl.load(V_block_ptr)
#         else:
#             v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
#         qk_wram = p.to(tl.float16)
#         qkv = tl.dot(tl.trans(qk_wram), v)
#         # -- update output accumulator --
#         acc = acc * alpha[:, None]
#         acc += qkv
#         # update m_i and l_i
#         m_i = m_ij
#         l_i = l_i * alpha + l_ij
#         V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
#         K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
 
#     return acc, l_i, m_i
 
 
# @triton.jit
# def _attn_eff_fwd_inner(
#         acc,
#         l_i,
#         m_i,
#         q,  #
#         K_block_ptr,
#         V_block_ptr,  #
#         start_m,
#         qk_scale,  #
#         Mask_block_ptr,
#         BLOCK_M: tl.constexpr,
#         BLOCK_DMODEL: tl.constexpr,
#         BLOCK_N: tl.constexpr,  #
#         STAGE: tl.constexpr,
#         offs_m: tl.constexpr,
#         offs_n: tl.constexpr,  #
#         N_CTX: tl.constexpr,
#         IS_DIVISIBLE: tl.constexpr,):
        
#     # causal = True
#     if STAGE == 1:
#         lo, hi = 0, start_m * BLOCK_M
#     elif STAGE == 2:
#         lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
#         lo = tl.multiple_of(lo, BLOCK_M)
#     # causal = False
#     else:
#         lo, hi = 0, N_CTX
#     K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
#     V_block_ptr = tl.advance(V_block_ptr, (0, lo))
#     Mask_block_ptr = tl.advance(Mask_block_ptr, (lo, 0))
#     # loop over k, v and update accumulator
#     for start_n in range(lo, hi, BLOCK_N):         # 处理 kv上的 (N_CTX/BLOCK_N) 数据
#         #tl.device_print("----- mask ----->\n")
#         start_n = tl.multiple_of(start_n, BLOCK_N)
#         # -- compute qk ----
#         if IS_DIVISIBLE:
#             k = tl.load(K_block_ptr)
#             mask = tl.load(Mask_block_ptr)
#         else:
#             k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
#             mask = tl.load(Mask_block_ptr, boundary_check=(0, 1), padding_option="zero")
#         qk = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
#         qk += tl.dot(k, q)
#         #tl.device_print("qk0:",qk)
        
#         qk = qk * qk_scale + mask*1.44269504
#         m_ij = tl.maximum(m_i, tl.max(qk, 0))
#         qk -= m_ij[None, :]
            
#         #tl.device_print("qk:",qk)
#         p = tl.exp2(qk)
#         l_ij = tl.sum(p, 0)
#         #tl.device_print("sum:",l_ij)
#         # -- update m_i and l_i
#         alpha = tl.exp2(m_i - m_ij)
#         if IS_DIVISIBLE:
#             v = tl.load(V_block_ptr)
#         else:
#             v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
#         #qkv = tl.dot(tl.trans(p.to(tl.float16)), v)
#         qk_wram = p.to(tl.float16)
#         qkv = tl.dot(tl.trans(qk_wram), v)
#         # -- update output accumulator --
#         acc = acc * alpha[:, None]
#         acc += qkv
#         # update m_i and l_i
#         m_i = m_ij
#         l_i = l_i * alpha + l_ij
#         V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
#         K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
#         Mask_block_ptr = tl.advance(Mask_block_ptr, (BLOCK_N, 0))
#     return acc, l_i, m_i 
 
# @triton.jit
# def _attn_eff_fwd(
#         Q,
#         K,
#         V,
#         sm_scale,
#         M,
#         Out,  #
#         stride_qz,
#         stride_qh,
#         stride_qm,
#         stride_qk,  #
#         stride_kz,
#         stride_kh,
#         stride_kn,
#         stride_kk,  #
#         stride_vz,
#         stride_vh,
#         stride_vk,
#         stride_vn,  #
#         stride_oz,
#         stride_oh,
#         stride_om,
#         stride_on,  #
#         stride_mm,
#         stride_mn,  #
#         Z,
#         H,  #
#         causal_mask,
#         N_CTX: tl.constexpr,  #
#         Q_N_CTX: tl.constexpr,
#         BLOCK_M: tl.constexpr,  #
#         BLOCK_N: tl.constexpr,  #
#         BLOCK_DMODEL: tl.constexpr,  #
#         STAGE: tl.constexpr,  #
#         IS_DIVISIBLE: tl.constexpr):
    
#     core_id = tl.program_id(0)
#     core_dim = tl.num_programs(0)
#     cluster_id = tl.program_id(1)
#     cluster_dim = tl.num_programs(1)

#     context_num = tl.cdiv(Q_N_CTX, BLOCK_M)
#     total_heads = Z * H
#     task_heads = total_heads // cluster_dim
#     task_remain_heads = total_heads % cluster_dim
#     task_heads += 1 if cluster_id < task_remain_heads else 0
#     task_head_begin = cluster_id * (total_heads // cluster_dim) + min(cluster_id, task_remain_heads)
#     if cluster_id >= task_remain_heads:
#         task_heads -= 1
#         task_head_begin = cluster_id * (total_heads // cluster_dim) + task_remain_heads
#     if task_heads <= 0:
#         return
 
#     core_heads = task_heads // core_dim
#     core_remain_heads = task_heads % core_dim
#     core_heads += 1 if core_id < core_remain_heads else 0
#     core_head_begin = core_id * (task_heads // core_dim) + min(core_id, core_remain_heads)
#     if core_id >= core_remain_heads:
#         core_heads -= 1
#         core_head_begin = core_id * (task_heads // core_dim) + core_remain_heads
#     if core_heads <= 0:
#         return
#     head_begin = task_head_begin + core_head_begin
#     head_end = head_begin + core_heads
 
#     for head_idx in range(head_begin, head_end):  # 一个core处理 q上的 (Q_N_CTX/BLOCK_M) 数据
#         start_m = (head_idx % context_num)
#         off_hz = head_idx
#         off_z = off_hz // H
#         off_h = off_hz % H
#         q_offset = (off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh)
#         kv_offset = (off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh)
#         # block pointers
#         Q_block_ptr = tl.make_block_ptr(
#             base=Q + q_offset,
#             shape=(BLOCK_DMODEL, Q_N_CTX),          
#             strides=(stride_qk, stride_qm),
#             offsets=(0, start_m * BLOCK_M),
#             block_shape=(BLOCK_DMODEL, BLOCK_M),
#             order=(0, 1),
#         )
#         K_block_ptr = tl.make_block_ptr(
#             base=K + kv_offset,
#             shape=(N_CTX, BLOCK_DMODEL),
#             strides=(stride_kn, stride_kk),
#             offsets=(0, 0),
#             block_shape=(BLOCK_N, BLOCK_DMODEL),
#             order=(1, 0),
#         )
        
#         V_block_ptr = tl.make_block_ptr(
#             base=V + kv_offset,
#             shape=(BLOCK_DMODEL, N_CTX),
#             strides=(stride_vk, stride_vn),
#             offsets=(0, 0),
#             block_shape=(BLOCK_DMODEL, BLOCK_N),
#             order=(0, 1),
#         )
     
#         Mask_block_ptr = tl.make_block_ptr(
#             base=causal_mask + off_z.to(tl.int64) * causal_mask.stride(0),
#             shape=(Q_N_CTX, N_CTX),
#             strides=(stride_mm, stride_mn),
#             offsets=(start_m * BLOCK_M, 0),
#             block_shape=(BLOCK_M, BLOCK_N),
#             order=(0, 1),
#         )

#         O_block_ptr = tl.make_block_ptr(
#             base=Out + q_offset,
#             shape=(Q_N_CTX, BLOCK_DMODEL),
#             strides=(stride_om, stride_on),
#             offsets=(start_m * BLOCK_M, 0),
#             block_shape=(BLOCK_M, BLOCK_DMODEL),
#             order=(0, 1),
#         )

#         # initialize offsets
#         offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
#         offs_n = tl.arange(0, BLOCK_N)
#         # initialize pointer to m and l
#         m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
#         l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
#         acc = tl.zeros([BLOCK_DMODEL, BLOCK_M], dtype=tl.float32)
#         # load scales
#         qk_scale = sm_scale
#         qk_scale *= 1.44269504  # 1/ln(2)
#         # load q: it will stay in SRAM throughout
#         if IS_DIVISIBLE:
#             q = tl.load(Q_block_ptr)
#         else:
#             q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
#         # stage 1: off-band
#         # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
#         # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
#         if STAGE & 1:
#             acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale, BLOCK_M, BLOCK_DMODEL, BLOCK_N, 1, offs_m, offs_n, N_CTX, IS_DIVISIBLE)
#         # stage 2: on-band
#         if STAGE & 2:
#             # barrier makes it easier for compielr to schedule the
#             # two loops independently
#             tl.debug_barrier()
#             acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale, BLOCK_M, BLOCK_DMODEL, BLOCK_N, 2, offs_m, offs_n, N_CTX, IS_DIVISIBLE)
#         # epilogue
#         m_i += tl.log2(l_i)
#         l_i_recip = 1.0 / l_i
#         acc = acc * l_i_recip[:, None]
#         acc = tl.trans(acc)
#         m_ptrs = M + off_hz * N_CTX + offs_m
#         if IS_DIVISIBLE:
#             tl.store(m_ptrs, m_i)
#             tl.store(O_block_ptr, acc.to(Out.type.element_ty))
#         else:
#             tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
#             tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))

# @triton.jit
# def _attn_bwd_preprocess(
#         O,
#         DO,  #
#         Delta,  #
#         Z,
#         H,
#         N_CTX,  #
#         BLOCK_M: tl.constexpr,
#         D_HEAD: tl.constexpr  #
# ):
#     off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
#     off_hz = tl.program_id(1)
#     off_n = tl.arange(0, D_HEAD)
#     # load
#     o = tl.load(O + off_hz * D_HEAD * N_CTX + off_m[:, None] * D_HEAD +
#                 off_n[None, :])
#     do = tl.load(DO + off_hz * D_HEAD * N_CTX + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
#     delta = tl.sum(o * do, axis=1)
#     # write-back
#     tl.store(Delta + off_hz * N_CTX + off_m, delta)
 
 
# # The main inner-loop logic for computing dK and dV.
# @triton.jit
# def _attn_bwd_dkdv(
#         dk,
#         dv,  #
#         Q,
#         k,
#         v,
#         sm_scale,  #
#         DO,  #
#         M,
#         D,  #
#         stride_tok,
#         stride_d,  #
#         H,
#         N_CTX,
#         BLOCK_M1: tl.constexpr,  #
#         BLOCK_N1: tl.constexpr,  #
#         BLOCK_DMODEL: tl.constexpr,  #
#         start_n,
#         start_m,
#         num_steps,  #
#         MASK: tl.constexpr):
#     offs_m = start_m + tl.arange(0, BLOCK_M1)
#     offs_n = start_n + tl.arange(0, BLOCK_N1)
#     offs_k = tl.arange(0, BLOCK_DMODEL)
#     qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
#     do_ptrs = DO + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
#     # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
#     tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
#     curr_m = start_m
#     step_m = BLOCK_M1
#     for blk_idx in range(num_steps):
#         qT = tl.load(qT_ptrs)
#         # Load m before computing qk to reduce pipeline stall.
#         offs_m = curr_m + tl.arange(0, BLOCK_M1)
#         m = tl.load(M + offs_m)
#         qkT = tl.dot(k, qT)
#         pT = tl.exp2(qkT - m[None, :])
#         # Autoregressive masking.
#         if MASK:
#             mask = (offs_m[None, :] >= offs_n[:, None])
#             pT = tl.where(mask, pT, 0.0)
#         do = tl.load(do_ptrs)
#         # Compute dV.
#         ppT = pT
#         ppT = ppT.to(tl.float16)
#         dv += tl.dot(ppT, do.to(tl.float16))
#         # D (= delta) is pre-divided by ds_scale.
#         Di = tl.load(D + offs_m)
#         # Compute dP and dS.
#         dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
#         dsT = pT * (dpT - Di[None, :])
#         dsT = dsT.to(tl.float16)
#         dk += tl.dot(dsT, tl.trans(qT))
#         # Increment pointers.
#         curr_m += step_m
#         qT_ptrs += BLOCK_M1 * stride_tok
#         do_ptrs += BLOCK_M1 * stride_tok
#     return dk, dv
 
 
# # the main inner-loop logic for computing dQ
# @triton.jit
# def _attn_bwd_dq(
#         dq,
#         q,
#         K,
#         V,  #
#         do,
#         m,
#         D,
#         stride_tok,
#         stride_d,  #
#         H,
#         N_CTX,  #
#         BLOCK_M2: tl.constexpr,  #
#         BLOCK_N2: tl.constexpr,  #
#         BLOCK_DMODEL: tl.constexpr,
#         # Filled in by the wrapper.
#         start_m,
#         start_n,
#         num_steps,  #
#         MASK: tl.constexpr):
#     offs_m = start_m + tl.arange(0, BLOCK_M2)
#     offs_n = start_n + tl.arange(0, BLOCK_N2)
#     offs_k = tl.arange(0, BLOCK_DMODEL)
#     kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
#     vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
#     # D (= delta) is pre-divided by ds_scale.
#     Di = tl.load(D + offs_m)
#     # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
#     tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
#     curr_n = start_n
#     step_n = BLOCK_N2
#     for blk_idx in range(num_steps):
#         kT = tl.load(kT_ptrs)
#         vT = tl.load(vT_ptrs)
#         qk = tl.dot(q, kT)
#         p = tl.exp2(qk - m[:, None])
#         # Autoregressive masking.
#         if MASK:
#             offs_n = curr_n + tl.arange(0, BLOCK_N2)
#             mask = (offs_m[:, None] >= offs_n[None, :])
#             p = tl.where(mask, p, 0.0)
#         # Compute dP and dS.
#         dp = tl.dot(do, tl.trans(vT)).to(tl.float32)
#         ds = p * (dp - Di[:, None])
#         ds = ds.to(tl.float16)
#         # Compute dQ.
#         # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
#         dq += tl.dot(ds, kT)
#         # Increment pointers.
#         curr_n += step_n
#         kT_ptrs += BLOCK_N2 * stride_tok
#         vT_ptrs += BLOCK_N2 * stride_tok
#     return dq   
 
 
# @triton.jit
# def _attn_bwd(
#         Q,
#         K,
#         V,
#         sm_scale,  #
#         DO,  #
#         DQ,
#         DK,
#         DV,  #
#         M,
#         D,
#         # shared by Q/K/V/DO.
#         stride_z,
#         stride_h,
#         stride_tok,
#         stride_d,  #
#         H,
#         N_CTX,  #
#         BLOCK_M1: tl.constexpr,  #
#         BLOCK_N1: tl.constexpr,  #
#         BLOCK_M2: tl.constexpr,  #
#         BLOCK_N2: tl.constexpr,  #
#         BLK_SLICE_FACTOR: tl.constexpr,  #
#         BLOCK_DMODEL: tl.constexpr):
#     LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
 
#     bhid = tl.program_id(2)
#     off_chz = bhid
#     adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
#     pid = tl.program_id(0)
 
#     # offset pointers for batch/head
#     Q += adj
#     K += adj
#     V += adj
#     DO += adj
#     DQ += adj
#     DK += adj
#     DV += adj
#     M += off_chz
#     D += off_chz
 
#     # load scales
#     offs_k = tl.arange(0, BLOCK_DMODEL)
 
#     start_n = pid * BLOCK_N1
#     start_m = start_n
 
#     MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
#     offs_n = start_n + tl.arange(0, BLOCK_N1)
    
#     dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
#     dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
 
#     # load K and V: they stay in SRAM throughout the inner loop.
#     k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
#     v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
 
#     num_steps = MASK_BLOCK_M1
 
#     dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D, stride_tok, stride_d, H, N_CTX, BLOCK_M1, BLOCK_N1, BLOCK_DMODEL, start_n, start_m, num_steps, MASK=True)
 
#     start_m += num_steps * BLOCK_M1
#     num_steps = (N_CTX - start_m) // BLOCK_M1
 
#     # Compute dK and dV for non-masked blocks.
#     dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D, stride_tok, stride_d, H, N_CTX, BLOCK_M1, BLOCK_N1, BLOCK_DMODEL, start_n, start_m, num_steps, MASK=False)
 
#     dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
#     tl.store(dv_ptrs, dv)
    
#     dk *= sm_scale
#     dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
#     tl.store(dk_ptrs, dk)
 
#     start_m = pid * BLOCK_M2
#     end_n = start_n + BLOCK_N1
 
#     MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
#     offs_m = start_m + tl.arange(0, BLOCK_M2)

#     q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
#     dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)
#     do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
 
#     m = tl.load(M + offs_m)
#     m = m[:, None]
 
#     # Compute dQ for masked (diagonal) blocks.
#     # NOTE: This code scans each row of QK^T backward (from right to left,
#     # but inside each call to _attn_bwd_dq, from left to right), but that's
#     # not due to anything important.  I just wanted to reuse the loop
#     # structure for dK & dV above as much as possible.
#     num_steps = MASK_BLOCK_N2
#     dq = _attn_bwd_dq(dq, q, K, V, do, m, D, stride_tok, stride_d, H, N_CTX, BLOCK_M2, BLOCK_N2, BLOCK_DMODEL, start_m, start_n, num_steps, MASK=True)
#     end_n -= num_steps * BLOCK_N2
#     # stage 2
#     num_steps = (end_n - start_n) // BLOCK_N2
#     dq = _attn_bwd_dq(dq, q, K, V, do, m, D, stride_tok, stride_d, H, N_CTX, BLOCK_M2, BLOCK_N2, BLOCK_DMODEL, start_m, start_n, num_steps, MASK=False)
#     # Write back dQ.
#     dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
#     dq *= LN2
#     tl.store(dq_ptrs, dq)
 
 
# empty = torch.empty(128, device="mlu")
 
 
# class _attention(torch.autograd.Function):
 
#     @staticmethod
#     def forward(ctx, q, k, v, causal, sm_scale,causal_mask):
#         # shape constraints
#         Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
#         assert Lq == Lk and Lk == Lv
#         #assert Lk in {16, 32, 64, 128}
#         o = torch.empty_like(q)
   
#         # 如果Nram不够，需要改为原始
#         # BLOCK_M = 128
#         # BLOCK_N = 64 if Lk <= 64 else 32
        
#         q_ctx_len = q.shape[-2]
#         kv_ctx_len = k.shape[-2]

#         if q_ctx_len <= 128:
#             if kv_ctx_len%128==0:
#                 BLOCK_M = q_ctx_len
#                 BLOCK_N = 128
#             else:
#                 BLOCK_M = q_ctx_len
#                 BLOCK_N = kv_ctx_len
#         elif q_ctx_len < 256:
#             if kv_ctx_len%64==0:
#                 BLOCK_M = q_ctx_len
#                 BLOCK_N = 64
#             else:
#                 BLOCK_M = q_ctx_len
#                 BLOCK_N = kv_ctx_len
#         elif q_ctx_len >= 256:
#             if kv_ctx_len%64==0:
#                 BLOCK_M = 64
#                 BLOCK_N = 64
#             elif kv_ctx_len%32==0:
#                 BLOCK_M = 32
#                 BLOCK_N = 32
#             else:
#                 BLOCK_M = 64
#                 BLOCK_N = kv_ctx_len
        
        


#         #print("------------BLOCK_M:",BLOCK_M)
#         #print("------------BLOCK_N:",BLOCK_N)


#         num_stages = 4 if Lk <= 64 else 3
#         num_warps = 1
#         stage = 3 if causal else 1
#         if torch.mlu.get_device_capability()[0] == 9:
#             num_warps = 8
#             # num_stages = 7 if Lk >= 64 else 3
#             num_stages = 0
#         num_stages = 0
#         #grid is coredim clusterdim 1
#         grid = (4, 8, 1)
#         M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

#         def is_divisible(a, b):
#             if b == 0:
#                 raise ValueError("Divisor cannot be 0")
#             return a % b == 0

#         N_CTX = q.shape[2]
#         IS_DIVISIBLE = False
#         if is_divisible(N_CTX, BLOCK_M) and is_divisible(N_CTX, BLOCK_N):
#             IS_DIVISIBLE = True


#         if(causal_mask is not None):
#             _attn_eff_fwd[grid](
#                 q,
#                 k,
#                 v,
#                 sm_scale,
#                 M,
#                 o,  #
#                 q.stride(0),
#                 q.stride(1),
#                 q.stride(2),
#                 q.stride(3),  #
#                 k.stride(0),
#                 k.stride(1),
#                 k.stride(2),
#                 k.stride(3),  #
#                 v.stride(0),
#                 v.stride(1),
#                 v.stride(2),
#                 v.stride(3),  #
#                 o.stride(0),
#                 o.stride(1),
#                 o.stride(2),
#                 o.stride(3),  #
#                 causal_mask.stride(2), 
#                 causal_mask.stride(3),
#                 q.shape[0],
#                 q.shape[1],  #
#                 causal_mask,
#                 N_CTX=k.shape[2],  #
#                 Q_N_CTX=q.shape[2],
#                 BLOCK_M=BLOCK_M,  #
#                 BLOCK_N=BLOCK_N,  #
#                 BLOCK_DMODEL=Lk,  # D_HEAD
#                 STAGE=stage,  #
#                 IS_DIVISIBLE=IS_DIVISIBLE,  #
#                 num_warps=num_warps,  #
#                 num_stages=num_stages  #
#             )
#         else:
#             _attn_fwd[grid](
#                 q,
#                 k,
#                 v,
#                 sm_scale,
#                 M,
#                 o,  #
#                 q.stride(0),
#                 q.stride(1),
#                 q.stride(2),
#                 q.stride(3),  #
#                 k.stride(0),
#                 k.stride(1),
#                 k.stride(2),
#                 k.stride(3),  #
#                 v.stride(0),
#                 v.stride(1),
#                 v.stride(2),
#                 v.stride(3),  #
#                 o.stride(0),
#                 o.stride(1),
#                 o.stride(2),
#                 o.stride(3),  #
#                 q.shape[0],
#                 q.shape[1],  #
#                 N_CTX=k.shape[2],  #
#                 Q_N_CTX=q.shape[2],
#                 BLOCK_M=BLOCK_M,  #
#                 BLOCK_N=BLOCK_N,  #
#                 BLOCK_DMODEL=Lk,  # D_HEAD
#                 STAGE=stage,  #
#                 IS_DIVISIBLE=IS_DIVISIBLE,  #
#                 num_warps=num_warps,  #
#                 num_stages=num_stages  #
#             )
 
 
 
#         ctx.save_for_backward(q, k, v, o, M)
#         ctx.grid = grid
#         ctx.sm_scale = sm_scale
#         ctx.BLOCK_DMODEL = Lk
#         ctx.causal = causal
#         return o
 
#     @staticmethod
#     def backward(ctx, do):
#         q, k, v, o, M = ctx.saved_tensors
#         do = do.contiguous()
#         assert (q.stride() == k.stride() == v.stride() == o.stride() == do.stride())
#         dq = torch.empty_like(q)
#         dk = torch.empty_like(k)
#         dv = torch.empty_like(v)
#         BATCH, N_HEAD, N_CTX = q.shape[:3]
#         PRE_BLOCK = 128
#         NUM_WARPS, NUM_STAGES = 4, 0
#         BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
#         BLK_SLICE_FACTOR = 2
#         RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
#         arg_k = k
#         arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
#         PRE_BLOCK = 128
#         #assert N_CTX % PRE_BLOCK == 0
#         pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
#         delta = torch.empty_like(M)
#         _attn_bwd_preprocess[pre_grid](o, do, delta, q.shape[0], q.shape[1], N_CTX, BLOCK_M=PRE_BLOCK, D_HEAD=q.shape[-1])
#         grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
#         _attn_bwd[grid](
#             q,
#             arg_k,
#             v,
#             ctx.sm_scale,
#             do,
#             dq,
#             dk,
#             dv,  #
#             M,
#             delta,  #
#             q.stride(0),
#             q.stride(1),
#             q.stride(2),
#             q.stride(3),  #
#             N_HEAD,
#             N_CTX,  #
#             BLOCK_M1=BLOCK_M1,
#             BLOCK_N1=BLOCK_N1,  #
#             BLOCK_M2=BLOCK_M2,
#             BLOCK_N2=BLOCK_N2,  #
#             BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
#             BLOCK_DMODEL=ctx.BLOCK_DMODEL,  #
#             num_warps=NUM_WARPS,  #
#             num_stages=NUM_STAGES  #
#         )
 
#         return dq, dk, dv, None, None
 
 
# attention = _attention.apply
 
# def test_op():
#     torch.manual_seed(20)
#     Z, H, N_CTX, D_HEAD=1,128,257,16
#     causal=False # 
#     causal_mask=[]
#     dtype=torch.float16
#     use_data_from_file=False
#     Q_N_CTX=N_CTX
    
#     for n in range(N_CTX,N_CTX+1):
#         if (use_data_from_file==False):
#             q = (torch.empty((Z, H, Q_N_CTX, D_HEAD), dtype=dtype, device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()).contiguous()
#             k = (torch.empty((Z, H, n, D_HEAD), dtype=dtype, device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()).contiguous()
#             v = (torch.empty((Z, H, n, D_HEAD), dtype=dtype,
#                          device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()).contiguous()
#             if(causal_mask is not None):
#                 causal_mask = (torch.empty((Z, 1, Q_N_CTX, n), dtype=dtype,
#                              device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()).contiguous()
#                 # causal_mask = (torch.zeros((Z, 1, Q_N_CTX, n), dtype=dtype,
#                              # device="mlu").requires_grad_()).contiguous()
#         else:
#             q_np = np.fromfile("query_states.npy", dtype=np.float16).reshape(Z, H, N_CTX, D_HEAD)
#             k_np = np.fromfile("key_states.npy", dtype=np.float16).reshape(Z, H, N_CTX, D_HEAD)
#             v_np = np.fromfile("value_states.npy", dtype=np.float16).reshape(Z, H, N_CTX, D_HEAD)

#             q = torch.from_numpy(q_np).to("mlu").reshape(Z, H, N_CTX, D_HEAD).requires_grad_()
#             k = torch.from_numpy(k_np).to("mlu").reshape(Z, H, N_CTX, D_HEAD).requires_grad_()
#             v = torch.from_numpy(v_np).to("mlu").reshape(Z, H, N_CTX, D_HEAD).requires_grad_()

#         sm_scale = 0.5
#         dout = torch.randn_like(q)
        
#         print("q:",q.shape)
#         print("k:",k.shape)
#         print("v:",v.shape)
#         print("causal:",causal)
        
#         # triton的实现
#         st=time.time()    
#         tri_out = attention(q, k, v, causal, sm_scale, causal_mask).to(torch.float16)
#         ed=time.time()
#         print("triton attention cost:",ed-st)
#         ##print("tri_out:",tri_out)
#         tri_out = tri_out.flatten()
#         nan_mask = torch.isnan(tri_out)
#         has_nan = torch.any(nan_mask)
#         #print("tri_out has_nan",has_nan)
        
        
#         # sdpa的实现
#         st=time.time()
#         if(causal_mask is not None): causal=False
#         sdpa_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, is_causal=causal, scale=sm_scale)
#         ed=time.time()
#         print("scaled_dot_product_attention attention cost:",ed-st)
#         ##print("sdpa_output:",sdpa_output)
#         sdpa_output = sdpa_output.flatten()
        
#         pytorch_valible=True
#         if(pytorch_valible==True):
#             ## pytorch的实现
#             st=time.time() 
#             M = torch.tril(torch.ones((Q_N_CTX, N_CTX), device="mlu"))
#             qk = torch.matmul(q, k.transpose(-2, -1))
#             p = qk * sm_scale
#             if(causal_mask is not None):
#                 p=p+causal_mask
#             elif causal:
#                 p[:, :, M == 0] = float("-inf")
            
#             if(1):
#                 p = torch.softmax(p.float(), dim=-1).to(torch.float16)
#                 pyt_out = torch.matmul(p, v)
#             pyt_out = pyt_out.flatten()
#             ed=time.time()
#             print("pytorch attention cost:",ed-st)
            
#             # compare
#             abs_tp_error = torch.sum(torch.abs(pyt_out - tri_out))
#             rel_tp_error = abs_tp_error / torch.minimum(torch.sum(torch.abs(pyt_out)), torch.sum(torch.abs(tri_out)))
#             print("abs_tp_error:",abs_tp_error)
#             print("rel_tp_error:",rel_tp_error)
#             abs_sp_error = torch.sum(torch.abs(pyt_out - sdpa_output))
#             rel_sp_error = abs_sp_error / torch.minimum(torch.sum(torch.abs(pyt_out)), torch.sum(torch.abs(sdpa_output)))
#             print("abs_sp_error:",abs_sp_error)
#             print("rel_sp_error:",rel_sp_error)
#         abs_ts_error = torch.sum(torch.abs(sdpa_output - tri_out))
#         rel_ts_error = abs_ts_error / torch.minimum(torch.sum(torch.abs(sdpa_output)), torch.sum(torch.abs(tri_out)))
#         print("abs_ts_error:",abs_ts_error)
#         print("rel_ts_error:",rel_ts_error)

 
# if __name__ == '__main__':
#     print("====================== Val =======================")
#     test_op()
     
#     #print("====================== Benchmark =======================")
#     #bench_flash_attention.run(save_path=".", print_data=True)

