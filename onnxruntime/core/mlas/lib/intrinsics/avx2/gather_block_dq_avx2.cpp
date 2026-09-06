/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    gather_block_dq_avx2.cpp

Abstract:

    This module implements the AVX2 kernels for the gather +
    block-dequantize fast path: one trailing-dim row slice of K quantized
    elements (Sym: 4-bit; Asym: QuantBits 2/4/8) in blocks of BlockSize.
    The op routes packed-unit-aligned rows (even K for 4-bit, K a multiple
    of 4 for 2-bit); the vector chunks additionally need a positive even
    BlockSize for 4-bit and a positive multiple of 4 for 2-bit (any positive
    value works for 8-bit). Layout and zero-point rules match the
    scalar baseline in gather_block_dq.cpp; the byte interleave happens
    BEFORE widening (dword interleave after widening would cross 128-bit
    lanes).

--*/

#include "../../mlasi.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

// Unpacks zero-point element (flat packed index base + b) and returns the
// signed (Sym=true) or raw value, or the packing default when packed_zp is
// null. Sym is 4-bit only; Asym dispatches on quant_bits at runtime
// (nibble-packed for 4-bit, quad-packed for 2-bit, one byte per block for
// 8-bit).
template <bool Sym>
inline int32_t
UnpackZp(const uint8_t* packed_zp, size_t base, size_t b, int32_t fallback, size_t quant_bits) {
  if (packed_zp == nullptr) {
    return fallback;
  }
  const size_t idx = base + b;
  if constexpr (Sym) {
    const uint8_t byte = packed_zp[idx >> 1];
    const int32_t nib = static_cast<int32_t>(((idx & 1) ? (byte >> 4) : byte) & 0x0F);
    return (nib ^ 8) - 8;
  } else {
    if (quant_bits == 8) {
      return static_cast<int32_t>(packed_zp[idx]);
    }
    if (quant_bits == 4) {
      const uint8_t byte = packed_zp[idx >> 1];
      return static_cast<int32_t>((((idx & 1) ? (byte >> 4) : byte) & 0x0F));
    }
    return static_cast<int32_t>((packed_zp[idx >> 2] >> ((idx & 3) * 2)) & 0x03);
  }
}

// Packed-byte index of element j for Bits (element 0 of each byte in the
// low bits; 8-bit rows are byte-per-element).
template <int Bits>
inline const uint8_t*
ElemBytes(const uint8_t* packed, size_t j) {
  if constexpr (Bits == 8) {
    return packed + j;
  } else if constexpr (Bits == 4) {
    return packed + (j >> 1);
  } else {
    return packed + (j >> 2);
  }
}

// Unpacks packed bytes into 32 ordered unsigned byte values: i0 holds
// elems 0..15, i1 holds elems 16..31. 4-bit: 16 packed bytes, nibble
// split; 2-bit: 8 packed bytes, quad split + 4-way byte interleave;
// 8-bit: 32 bytes, split 256-bit load into halves.
template <int Bits>
inline void
Unpack32(const uint8_t* packed, __m128i& i0, __m128i& i1) {
  if constexpr (Bits == 8) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));
    i0 = _mm256_castsi256_si128(v);
    i1 = _mm256_extracti128_si256(v, 1);
  } else if constexpr (Bits == 4) {
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const __m128i lowmask = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(b, lowmask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), lowmask);
    i0 = _mm_unpacklo_epi8(lo, hi);
    i1 = _mm_unpackhi_epi8(lo, hi);
  } else {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i_u*>(packed));
    const __m128i lowmask = _mm_set1_epi8(0x03);
    const __m128i lo = _mm_and_si128(b, lowmask);
    // Right shifts bleed neighboring-byte bits into the top of each byte,
    // but the low mask keeps only the wanted field, so a 16-bit lane shift
    // is exact here (same argument as the 4-bit nibble split).
    const __m128i m1 = _mm_and_si128(_mm_srli_epi16(b, 2), lowmask);
    const __m128i m2 = _mm_and_si128(_mm_srli_epi16(b, 4), lowmask);
    const __m128i m3 = _mm_and_si128(_mm_srli_epi16(b, 6), lowmask);
    const __m128i ab_lo = _mm_unpacklo_epi8(lo, m1);
    const __m128i cd_lo = _mm_unpacklo_epi8(m2, m3);
    i0 = _mm_unpacklo_epi16(ab_lo, cd_lo);
    i1 = _mm_unpackhi_epi16(ab_lo, cd_lo);
  }
}

// Unpacks packed bytes into 16 ordered unsigned byte values in i0
// (elems 0..15). 4-bit: 8 packed bytes (8-byte load, no row-end over-read);
// 2-bit: 4 packed bytes (32-bit load, upper xmm bytes zeroed, only the low
// halves feed the interleave); 8-bit: 16 bytes.
template <int Bits>
inline void
Unpack16(const uint8_t* packed, __m128i& i0) {
  if constexpr (Bits == 8) {
    i0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
  } else if constexpr (Bits == 4) {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i_u*>(packed));
    const __m128i lowmask = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(b, lowmask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), lowmask);
    i0 = _mm_unpacklo_epi8(lo, hi);
  } else {
    uint32_t word;
    memcpy(&word, packed, sizeof(word));
    const __m128i b = _mm_cvtsi32_si128(static_cast<int>(word));
    const __m128i lowmask = _mm_set1_epi8(0x03);
    const __m128i lo = _mm_and_si128(b, lowmask);
    const __m128i m1 = _mm_and_si128(_mm_srli_epi16(b, 2), lowmask);
    const __m128i m2 = _mm_and_si128(_mm_srli_epi16(b, 4), lowmask);
    const __m128i m3 = _mm_and_si128(_mm_srli_epi16(b, 6), lowmask);
    const __m128i ab_lo = _mm_unpacklo_epi8(lo, m1);
    const __m128i cd_lo = _mm_unpacklo_epi8(m2, m3);
    i0 = _mm_unpacklo_epi16(ab_lo, cd_lo);
  }
}

// Widens the low 8 bytes of v to float and subtracts zp (signed widen for
// Sym, unsigned for Asym).
template <bool Sym>
inline __m256
Cvt8(__m128i v, __m256i zp) {
  __m256i q;
  if constexpr (Sym) {
    q = _mm256_cvtepi8_epi32(v);
  } else {
    q = _mm256_cvtepu8_epi32(v);
  }
  return _mm256_cvtepi32_ps(_mm256_sub_epi32(q, zp));
}

// Dequantizes exactly 32 elements into 32 floats. Sign extension (Sym,
// 4-bit only) is per byte so it commutes with the interleave and happens
// after unpacking.
template <bool Sym, int Bits>
inline void
DequantChunk32(const uint8_t* packed, __m256 scale, __m256i zp, float* out) {
  static_assert(!Sym || Bits == 4, "Sym entries are 4-bit only");
  __m128i i0, i1;
  Unpack32<Bits>(packed, i0, i1);
  if constexpr (Sym) {
    const __m128i sign = _mm_set1_epi8(0x08);
    i0 = _mm_sub_epi8(_mm_xor_si128(i0, sign), sign);
    i1 = _mm_sub_epi8(_mm_xor_si128(i1, sign), sign);
  }
  const __m256 f0 = Cvt8<Sym>(i0, zp);
  const __m256 f1 = Cvt8<Sym>(_mm_srli_si128(i0, 8), zp);
  const __m256 f2 = Cvt8<Sym>(i1, zp);
  const __m256 f3 = Cvt8<Sym>(_mm_srli_si128(i1, 8), zp);
  _mm256_storeu_ps(out + 0, _mm256_mul_ps(f0, scale));
  _mm256_storeu_ps(out + 8, _mm256_mul_ps(f1, scale));
  _mm256_storeu_ps(out + 16, _mm256_mul_ps(f2, scale));
  _mm256_storeu_ps(out + 24, _mm256_mul_ps(f3, scale));
}

// Dequantizes exactly 16 elements into 16 floats.
template <bool Sym, int Bits>
inline void
DequantChunk16(const uint8_t* packed, __m256 scale, __m256i zp, float* out) {
  static_assert(!Sym || Bits == 4, "Sym entries are 4-bit only");
  __m128i i0;
  Unpack16<Bits>(packed, i0);
  if constexpr (Sym) {
    const __m128i sign = _mm_set1_epi8(0x08);
    i0 = _mm_sub_epi8(_mm_xor_si128(i0, sign), sign);
  }
  const __m256 f0 = Cvt8<Sym>(i0, zp);
  const __m256 f1 = Cvt8<Sym>(_mm_srli_si128(i0, 8), zp);
  _mm256_storeu_ps(out + 0, _mm256_mul_ps(f0, scale));
  _mm256_storeu_ps(out + 8, _mm256_mul_ps(f1, scale));
}

// Scalar tail for remainders below the vector widths. The packed base
// advances with j, which keeps the block start's alignment, so the
// relative sub-byte indexing below matches the chunk paths.
template <bool Sym, int Bits>
inline void
DequantChunkScalar(const uint8_t* packed, float scale, int32_t zp, float* out, size_t count) {
  static_assert(!Sym || Bits == 4, "Sym entries are 4-bit only");
  for (size_t j = 0; j < count; ++j) {
    int32_t q;
    if constexpr (Bits == 8) {
      q = static_cast<int32_t>(packed[j]);
    } else if constexpr (Bits == 4) {
      const uint8_t byte = packed[j >> 1];
      q = static_cast<int32_t>(((j & 1) ? (byte >> 4) : byte) & 0x0F);
      if constexpr (Sym) {
        q = (q ^ 8) - 8;
      }
    } else {
      const uint8_t byte = packed[j >> 2];
      q = static_cast<int32_t>((byte >> ((j & 3) * 2)) & 0x03);
    }
    out[j] = static_cast<float>(q - zp) * scale;
  }
}

inline void
Store32Fp16(const float* tmp, uint16_t* out) {
  _mm_storeu_si128(
      reinterpret_cast<__m128i*>(out + 0),
      _mm256_cvtps_ph(_mm256_loadu_ps(tmp + 0), _MM_FROUND_TO_NEAREST_INT));
  _mm_storeu_si128(
      reinterpret_cast<__m128i*>(out + 8),
      _mm256_cvtps_ph(_mm256_loadu_ps(tmp + 8), _MM_FROUND_TO_NEAREST_INT));
  _mm_storeu_si128(
      reinterpret_cast<__m128i*>(out + 16),
      _mm256_cvtps_ph(_mm256_loadu_ps(tmp + 16), _MM_FROUND_TO_NEAREST_INT));
  _mm_storeu_si128(
      reinterpret_cast<__m128i*>(out + 24),
      _mm256_cvtps_ph(_mm256_loadu_ps(tmp + 24), _MM_FROUND_TO_NEAREST_INT));
}

inline void
Store16Fp16(const float* tmp, uint16_t* out) {
  _mm_storeu_si128(
      reinterpret_cast<__m128i*>(out + 0),
      _mm256_cvtps_ph(_mm256_loadu_ps(tmp + 0), _MM_FROUND_TO_NEAREST_INT));
  _mm_storeu_si128(
      reinterpret_cast<__m128i*>(out + 8),
      _mm256_cvtps_ph(_mm256_loadu_ps(tmp + 8), _MM_FROUND_TO_NEAREST_INT));
}

// Float-output row kernel: vector stores go straight to the output.
// quant_bits is runtime (2, 4, or 8 on the Asym path; Sym is always 4)
// and selects the instantiation via the entry-point switch below.
template <bool Sym, int Bits>
void
DequantizeRowFloat(const uint8_t* packed_row, const float* scales, const uint8_t* packed_zp,
                   size_t zp_base, size_t k, size_t block_size, float* out, int32_t zp_fallback,
                   size_t quant_bits) {
  const size_t num_blocks = (k + block_size - 1) / block_size;
  for (size_t b = 0; b < num_blocks; ++b) {
    const float scale = scales[b];
    const __m256 scale_vec = _mm256_broadcast_ss(&scale);
    const int32_t zp = UnpackZp<Sym>(packed_zp, zp_base, b, zp_fallback, quant_bits);
    const __m256i zp_vec = _mm256_set1_epi32(zp);
    const size_t j0 = b * block_size;
    size_t j1 = j0 + block_size;
    if (j1 > k) {
      j1 = k;
    }
    size_t j = j0;
    for (; j + 32 <= j1; j += 32) {
      DequantChunk32<Sym, Bits>(ElemBytes<Bits>(packed_row, j), scale_vec, zp_vec, out + j);
    }
    if (j1 - j == 16) {
      DequantChunk16<Sym, Bits>(ElemBytes<Bits>(packed_row, j), scale_vec, zp_vec, out + j);
      j += 16;
    }
    if (j < j1) {
      DequantChunkScalar<Sym, Bits>(ElemBytes<Bits>(packed_row, j), scale, zp, out + j, j1 - j);
    }
  }
}

// Fp16-output row kernel: vector chunks convert through a 32-float scratch.
template <bool Sym, int Bits>
void
DequantizeRowFp16(const uint8_t* packed_row, const float* scales, const uint8_t* packed_zp,
                  size_t zp_base, size_t k, size_t block_size, uint16_t* out, int32_t zp_fallback,
                  size_t quant_bits) {
  const size_t num_blocks = (k + block_size - 1) / block_size;
  alignas(32) float tmp[32];
  for (size_t b = 0; b < num_blocks; ++b) {
    const float scale = scales[b];
    const __m256 scale_vec = _mm256_broadcast_ss(&scale);
    const int32_t zp = UnpackZp<Sym>(packed_zp, zp_base, b, zp_fallback, quant_bits);
    const __m256i zp_vec = _mm256_set1_epi32(zp);
    const size_t j0 = b * block_size;
    size_t j1 = j0 + block_size;
    if (j1 > k) {
      j1 = k;
    }
    size_t j = j0;
    for (; j + 32 <= j1; j += 32) {
      DequantChunk32<Sym, Bits>(ElemBytes<Bits>(packed_row, j), scale_vec, zp_vec, tmp);
      Store32Fp16(tmp, out + j);
    }
    if (j1 - j == 16) {
      DequantChunk16<Sym, Bits>(ElemBytes<Bits>(packed_row, j), scale_vec, zp_vec, tmp);
      Store16Fp16(tmp, out + j);
      j += 16;
    }
    if (j < j1) {
      DequantChunkScalar<Sym, Bits>(ElemBytes<Bits>(packed_row, j), scale, zp, tmp, j1 - j);
      for (size_t t = 0; t < j1 - j; ++t) {
        out[j + t] = onnxruntime::MLFloat16(tmp[t]).val;
      }
    }
  }
}

}  // namespace

void
MLASCALL
MlasGatherBlockDequantizeSymKernelAvx2(
    float* Output,
    const uint8_t* PackedRow,
    const float* Scales,
    const uint8_t* PackedZeroPoints,
    size_t ZeroPointBase,
    size_t K,
    size_t BlockSize) {
  DequantizeRowFloat<true, 4>(PackedRow, Scales, PackedZeroPoints, ZeroPointBase, K, BlockSize, Output,
                              0, 4);
}

void
MLASCALL
MlasGatherBlockDequantizeSymFp16KernelAvx2(
    uint16_t* Output,
    const uint8_t* PackedRow,
    const float* Scales,
    const uint8_t* PackedZeroPoints,
    size_t ZeroPointBase,
    size_t K,
    size_t BlockSize) {
  DequantizeRowFp16<true, 4>(PackedRow, Scales, PackedZeroPoints, ZeroPointBase, K, BlockSize, Output,
                             0, 4);
}

void
MLASCALL
MlasGatherBlockDequantizeAsymKernelAvx2(
    float* Output,
    const uint8_t* PackedRow,
    const float* Scales,
    const uint8_t* PackedZeroPoints,
    size_t ZeroPointBase,
    int32_t DefaultZeroPoint,
    size_t QuantBits,
    size_t K,
    size_t BlockSize) {
  if (QuantBits == 2) {
    DequantizeRowFloat<false, 2>(PackedRow, Scales, PackedZeroPoints, ZeroPointBase, K, BlockSize,
                                 Output, DefaultZeroPoint, QuantBits);
  } else if (QuantBits == 8) {
    DequantizeRowFloat<false, 8>(PackedRow, Scales, PackedZeroPoints, ZeroPointBase, K, BlockSize,
                                 Output, DefaultZeroPoint, QuantBits);
  } else {
    DequantizeRowFloat<false, 4>(PackedRow, Scales, PackedZeroPoints, ZeroPointBase, K, BlockSize,
                                 Output, DefaultZeroPoint, QuantBits);
  }
}

void
MLASCALL
MlasGatherBlockDequantizeAsymFp16KernelAvx2(
    uint16_t* Output,
    const uint8_t* PackedRow,
    const float* Scales,
    const uint8_t* PackedZeroPoints,
    size_t ZeroPointBase,
    int32_t DefaultZeroPoint,
    size_t QuantBits,
    size_t K,
    size_t BlockSize) {
  if (QuantBits == 2) {
    DequantizeRowFp16<false, 2>(PackedRow, Scales, PackedZeroPoints, ZeroPointBase, K, BlockSize,
                                Output, DefaultZeroPoint, QuantBits);
  } else if (QuantBits == 8) {
    DequantizeRowFp16<false, 8>(PackedRow, Scales, PackedZeroPoints, ZeroPointBase, K, BlockSize,
                                Output, DefaultZeroPoint, QuantBits);
  } else {
    DequantizeRowFp16<false, 4>(PackedRow, Scales, PackedZeroPoints, ZeroPointBase, K, BlockSize,
                                Output, DefaultZeroPoint, QuantBits);
  }
}
