/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    gather_block_dq_avx512f.cpp

Abstract:

    This module implements the AVX512F kernels for the gather +
    block-dequantize fast path: one trailing-dim row slice of K 4-bit
    elements (K even) in blocks of BlockSize. Layout and zero-point rules
    match the scalar baseline in gather_block_dq.cpp. Only FP32-output
    entries live here; FP16 rows stay on the AVX2 tier (dispatch never
    selects this TU for FP16). Asym entries take QuantBits but only
    accelerate 4-bit here; other widths delegate to the AVX2 entry
    (strictly faster than scalar, and AVX512F implies AVX2).

    Compute core (AVX512F only, no BW/VL): one 64-element iteration =
    32 packed bytes. The zmm load splits into two 128-bit halves; each half
    reuses the SSE2-class nibble split + byte interleave, giving 16 ordered
    nibbles per shot. Dequant is a per-block 16-entry float LUT built in the
    int domain (lane i holds sext4(i)-zp exactly, then one int->float
    convert and one scale multiply), indexed by zero-extended nibbles via
    _mm512_permutexvar_ps: no per-element sign extension, no per-element
    int->float converts, no per-element subtract/multiply. A standalone
    benchmark (K=768) picked the LUT over the arithmetic halves-kernel:
    1.15-1.63x faster symmetric, 1.02-1.27x asymmetric at block >= 32,
    parity at asymmetric block 16. Remainders below 64 elements go through
    a scalar tail.

--*/

#include "../../mlasi.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace {

// Constant lane ids 15..0 and symmetric sign-extended nibble values: lane i
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
inline void
Unpack64(const uint8_t* packed, __m128i shots[4]) {
  const __m512i v = _mm512_loadu_si512(static_cast<const void*>(packed));
  const __m128i halves[2] = {_mm512_castsi512_si128(v), _mm512_extracti32x4_epi32(v, 1)};
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
  if (QuantBits != 4) {
    // Only 4-bit rows use the AVX512F core; other widths delegate to the
    // AVX2 entry (same library, bit-identical by construction).
    MlasGatherBlockDequantizeAsymKernelAvx2(Output, PackedRow, Scales, PackedZeroPoints,
                                            ZeroPointBase, DefaultZeroPoint, QuantBits, K,
                                            BlockSize);
    return;
  }
  DequantizeRowFloat<false>(PackedRow, Scales, PackedZeroPoints, ZeroPointBase, K, BlockSize,
                            Output, DefaultZeroPoint);
}
