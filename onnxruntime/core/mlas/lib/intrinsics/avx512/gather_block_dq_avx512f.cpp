/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    gather_block_dq_avx512f.cpp

Abstract:

    This module implements the AVX512F kernels for the gather +
    block-dequantize fast path: one trailing-dim row slice of K 4-bit or
    8-bit elements in blocks of BlockSize. Layout and zero-point rules
    match the scalar baseline in gather_block_dq.cpp. Only FP32-output
    entries live here; FP16 rows stay on the AVX2 tier (dispatch never
    selects this TU for FP16). Asym entries take QuantBits: 4-bit rows use
    the LUT core and 8-bit rows the arithmetic core below; 2-bit rows
    delegate to the AVX2 entry (5-17x faster than scalar across blocks
    16-128 in the in-tree row benchmark, and AVX512F implies AVX2).
    BlockSize must be a positive multiple of 16 (the
    16-element shots assume block-aligned shots); the op enforces a power
    of 2 >= 16.

    Compute core (AVX512F only, no BW/VL): one 64-element iteration =
    32 packed bytes fetched with two 128-bit loads (a full 64-byte zmm
    load would over-read the row on the last step). Each half reuses the
    SSE2-class nibble split + byte interleave, giving 16 ordered nibbles
    per shot. Dequant is a per-block 16-entry float LUT built in the
    int domain (lane i holds sext4(i)-zp exactly, then one int->float
    convert and one scale multiply), indexed by zero-extended nibbles via
    _mm512_permutexvar_ps: no per-element sign extension, no per-element
    int->float converts, no per-element subtract/multiply. The in-tree row
    benchmark (K=768, BM_GatherBlockDequantizeRow) shows the LUT core ahead
    of the AVX2 path by 1.7-2.7x symmetric and 1.6-2.4x asymmetric across
    blocks 16-128, with and without an explicit zero-point tensor.
    Remainders below 64 elements go through a scalar tail.

    8-bit rows use a separate arithmetic core (plain byte loads, widen +
    subtract + convert + multiply per 16-elem shot, same block counter):
    a 256-entry table would cost 16 zmm loads per block before any compute,
    against 4 ALU ops per shot here. The in-tree row benchmark (K=768) has
    the core ahead of the AVX2 8-bit path by 1.3-1.8x at block <= 32 and
    ~1.2x at block 64, reaching parity at block 128 — so block >= 128
    delegates to AVX2.

--*/

#include "../../mlasi.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace {

// Constant lane ids 0..15 and symmetric sign-extended nibble values: lane i
// holds sext4(i) exactly in the int domain.
alignas(64) const int32_t LaneIds[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
alignas(64) const int32_t SymSext[16] = {0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};

// Unpacks zero-point nibble (flat packed index) and returns the signed
// (Sym=true) or raw value, or the packing default when packed_zp is null.
template <bool Sym>
inline int32_t
UnpackZpNibble(const uint8_t* packed_zp, size_t base, size_t b, int32_t fallback) {
  if (packed_zp == nullptr) {
    return fallback;
  }
  const size_t idx = base + b;
  const uint8_t byte = packed_zp[idx >> 1];
  const int32_t nib = static_cast<int32_t>(((idx & 1) ? (byte >> 4) : byte) & 0x0F);
  if constexpr (Sym) {
    return (nib ^ 8) - 8;
  } else {
    return nib;
  }
}

// Unpacks 32 packed bytes into four shots of 16 ordered nibble values
// (shots cover elems 0..15, 16..31, 32..47, 48..63 of the 64-element step).
// Two 128-bit loads read exactly the 32 bytes the step consumes; a full
// 64-byte zmm load would over-read the row on the last step.
inline void
Unpack64(const uint8_t* packed, __m128i shots[4]) {
  const __m128i halves[2] = {_mm_loadu_si128(reinterpret_cast<const __m128i*>(packed)),
                             _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + 16))};
  const __m128i lowmask = _mm_set1_epi8(0x0F);
  for (int h = 0; h < 2; ++h) {
    const __m128i lo = _mm_and_si128(halves[h], lowmask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(halves[h], 4), lowmask);
    shots[2 * h + 0] = _mm_unpacklo_epi8(lo, hi);
    shots[2 * h + 1] = _mm_unpackhi_epi8(lo, hi);
  }
}

// Builds the 16-entry float LUT for one block: LUT[i] = (base[i] - zp) *
// scale with base = sext4 (Sym) or identity (Asym), all exact in int. The
// unaligned zmm load is AVX512F (VMOVDQU32); the tables are cache-resident.
template <bool Sym>
inline __m512
BuildLut(float scale, int32_t zp) {
  const __m512i base = _mm512_loadu_si512(static_cast<const void*>(Sym ? SymSext : LaneIds));
  const __m512i shifted = _mm512_sub_epi32(base, _mm512_set1_epi32(zp));
  return _mm512_mul_ps(_mm512_cvtepi32_ps(shifted), _mm512_set1_ps(scale));
}

// Scalar tail for remainders below 64 elements.
template <bool Sym>
inline void
DequantTail(const uint8_t* packed_row, const float* scales, const uint8_t* packed_zp,
            size_t zp_base, size_t j0, size_t k, size_t block_size, int32_t zp_fallback,
            float* out) {
  for (size_t j = j0; j < k; ++j) {
    const size_t b = j / block_size;
    int32_t zp = UnpackZpNibble<Sym>(packed_zp, zp_base, b, zp_fallback);
    const uint8_t byte = packed_row[j >> 1];
    int32_t q = static_cast<int32_t>(((j & 1) ? (byte >> 4) : byte) & 0x0F);
    if constexpr (Sym) {
      q = (q ^ 8) - 8;
    }
    out[j] = static_cast<float>(q - zp) * scales[b];
  }
}

// Float-output row kernel: 64-element LUT steps, then the scalar tail.
template <bool Sym>
void
DequantizeRowFloat(const uint8_t* packed_row, const float* scales, const uint8_t* packed_zp,
                   size_t zp_base, size_t k, size_t block_size, float* out, int32_t zp_fallback) {
  const size_t shots_per_block = block_size >> 4;
  size_t b = 0;
  size_t shot_in_block = shots_per_block;  // force LUT build on first shot
  __m512 lut = _mm512_setzero_ps();
  size_t j = 0;
  for (; j + 64 <= k; j += 64) {
    __m128i shots[4];
    Unpack64(packed_row + (j >> 1), shots);
    for (int s = 0; s < 4; ++s) {
      if (shot_in_block == shots_per_block) {
        shot_in_block = 0;
        lut = BuildLut<Sym>(scales[b], UnpackZpNibble<Sym>(packed_zp, zp_base, b, zp_fallback));
        ++b;
      }
      ++shot_in_block;
      const __m512i idx = _mm512_cvtepu8_epi32(shots[s]);
      _mm512_storeu_ps(out + j + static_cast<size_t>(16 * s), _mm512_permutexvar_ps(idx, lut));
    }
  }
  DequantTail<Sym>(packed_row, scales, packed_zp, zp_base, j, k, block_size, zp_fallback, out);
}

// 8-bit float-output row kernel: values and zero points are plain bytes
// (one zero point per block, DefaultZeroPoint when packed_zp is null).
// 64-element steps from one zmm load, four 16-elem widen shots across the
// 128-bit lanes, per-block scale/zp via the shot counter. Asym-only: Sym
// entries are 4-bit only, so no Sym instantiation exists.
void
DequantizeRowFloatBits8(const uint8_t* packed_row, const float* scales, const uint8_t* packed_zp,
                        size_t zp_base, size_t k, size_t block_size, float* out,
                        int32_t zp_fallback) {
  const size_t shots_per_block = block_size >> 4;
  size_t b = 0;
  size_t shot_in_block = shots_per_block;  // force load on first shot
  __m512 scale_vec = _mm512_setzero_ps();
  __m512i zp_vec = _mm512_setzero_si512();
  size_t j = 0;
  for (; j + 64 <= k; j += 64) {
    const __m512i v = _mm512_loadu_si512(static_cast<const void*>(packed_row + j));
    const __m128i lanes[4] = {_mm512_castsi512_si128(v), _mm512_extracti32x4_epi32(v, 1),
                              _mm512_extracti32x4_epi32(v, 2), _mm512_extracti32x4_epi32(v, 3)};
    for (int s = 0; s < 4; ++s) {
      if (shot_in_block == shots_per_block) {
        shot_in_block = 0;
        scale_vec = _mm512_set1_ps(scales[b]);
        const int32_t zp =
            packed_zp == nullptr ? zp_fallback : static_cast<int32_t>(packed_zp[zp_base + b]);
        zp_vec = _mm512_set1_epi32(zp);
        ++b;
      }
      ++shot_in_block;
      const __m512i q = _mm512_cvtepu8_epi32(lanes[s]);
      const __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(q, zp_vec)), scale_vec);
      _mm512_storeu_ps(out + j + static_cast<size_t>(16 * s), f);
    }
  }
  for (; j < k; ++j) {
    const size_t bb = j / block_size;
    const int32_t zp =
        packed_zp == nullptr ? zp_fallback : static_cast<int32_t>(packed_zp[zp_base + bb]);
    out[j] = static_cast<float>(static_cast<int32_t>(packed_row[j]) - zp) * scales[bb];
  }
}

}  // namespace

void
MLASCALL
MlasGatherBlockDequantizeSymKernelAvx512F(
    float* Output,
    const uint8_t* PackedRow,
    const float* Scales,
    const uint8_t* PackedZeroPoints,
    size_t ZeroPointBase,
    size_t K,
    size_t BlockSize) {
  DequantizeRowFloat<true>(PackedRow, Scales, PackedZeroPoints, ZeroPointBase, K, BlockSize, Output,
                           0);
}

void
MLASCALL
MlasGatherBlockDequantizeAsymKernelAvx512F(
    float* Output,
    const uint8_t* PackedRow,
    const float* Scales,
    const uint8_t* PackedZeroPoints,
    size_t ZeroPointBase,
    int32_t DefaultZeroPoint,
    size_t QuantBits,
    size_t K,
    size_t BlockSize) {
  if (QuantBits == 8) {
    // 8-bit arithmetic core, except at block >= 128 where it ties the AVX2
    // entry within noise — delegate there so no cell regresses.
    if (BlockSize >= 128) {
      MlasGatherBlockDequantizeAsymKernelAvx2(Output, PackedRow, Scales, PackedZeroPoints,
                                              ZeroPointBase, DefaultZeroPoint, QuantBits, K,
                                              BlockSize);
      return;
    }
    DequantizeRowFloatBits8(PackedRow, Scales, PackedZeroPoints, ZeroPointBase, K, BlockSize,
                            Output, DefaultZeroPoint);
    return;
  }
  if (QuantBits != 4) {
    // Only 4-bit rows use the AVX512F LUT core; 2-bit rows delegate to the
    // AVX2 entry (same library, bit-identical by construction).
    MlasGatherBlockDequantizeAsymKernelAvx2(Output, PackedRow, Scales, PackedZeroPoints,
                                            ZeroPointBase, DefaultZeroPoint, QuantBits, K,
                                            BlockSize);
    return;
  }
  DequantizeRowFloat<false>(PackedRow, Scales, PackedZeroPoints, ZeroPointBase, K, BlockSize,
                            Output, DefaultZeroPoint);
}
