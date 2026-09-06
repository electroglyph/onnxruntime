/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    gather_block_dq.cpp

Abstract:

    This module implements the scalar baseline kernels for the gather +
    block-dequantize fast path: one trailing-dim row slice of K quantized
    elements in blocks of BlockSize. Sym entries are 4-bit only (see the
    nibble layout below); Asym entries take QuantBits (2, 4, or 8). See the
    matching vector kernels in intrinsics/avx2/gather_block_dq_avx2.cpp
    and intrinsics/avx512/gather_block_dq_avx512f.cpp.

    Nibble layout (both 4-bit packings): the even element lives in the low
    nibble. Zero points are one nibble per block starting at nibble
    ZeroPointBase (flat packed index); a null pointer means the packing
    default (0 for Sym; caller-picked DefaultZeroPoint for Asym). The op
    only routes packed-unit-aligned rows here (even K for 4-bit, K a
    multiple of 4 for 2-bit; 8-bit rows are always byte aligned) and uses
    the generic op loop otherwise. 2-bit values pack four per byte
    (element 0 in the low 2 bits) with the zero points packed the same way;
    8-bit values and zero points are plain bytes.

--*/

#include "mlasi.h"

#include <cstddef>
#include <cstdint>

namespace {

// Extracts nibble idx (0-based, low nibble first) from flat packed bytes.
// Sym path is 4-bit only.
inline uint8_t
UnpackNibble(const uint8_t* packed, size_t idx) {
  const uint8_t byte = packed[idx >> 1];
  return (idx & 1) ? static_cast<uint8_t>((byte >> 4) & 0x0F) : static_cast<uint8_t>(byte & 0x0F);
}

// Extracts element idx for QuantBits in {2, 4, 8} (element 0 of each byte
// in the low bits). Matches Get2BitElementUint8/Get4BitElement<uint8_t> and
// plain byte loads in the contrib op.
inline uint8_t
UnpackValue(const uint8_t* packed, size_t idx, size_t quant_bits) {
  if (quant_bits == 8) {
    return packed[idx];
  }
  const uint8_t byte = packed[idx >> (quant_bits == 4 ? 1 : 2)];
  const unsigned shift = static_cast<unsigned>((idx & (quant_bits == 4 ? 1 : 3)) * quant_bits);
  return static_cast<uint8_t>((byte >> shift) & ((1u << quant_bits) - 1));
}

// Extracts zero-point element idx for QuantBits in {2, 4, 8}: sub-byte
// packed like the data for 2/4-bit, one byte per block for 8-bit.
inline int32_t
UnpackZpValue(const uint8_t* packed_zp, size_t idx, size_t quant_bits) {
  if (quant_bits == 8) {
    return static_cast<int32_t>(packed_zp[idx]);
  }
  return static_cast<int32_t>(UnpackValue(packed_zp, idx, quant_bits));
}

}  // namespace

void
MLASCALL
MlasGatherBlockDequantizeSymKernel(
    float* Output,
    const uint8_t* PackedRow,
    const float* Scales,
    const uint8_t* PackedZeroPoints,
    size_t ZeroPointBase,
    size_t K,
    size_t BlockSize) {
  const size_t num_blocks = (K + BlockSize - 1) / BlockSize;
  for (size_t b = 0; b < num_blocks; ++b) {
    const float scale = Scales[b];
    int32_t zp = 0;
    if (PackedZeroPoints != nullptr) {
      zp = static_cast<int32_t>(UnpackNibble(PackedZeroPoints, ZeroPointBase + b));
      zp = (zp ^ 8) - 8;
    }
    size_t j_end = (b + 1) * BlockSize;
    if (j_end > K) {
      j_end = K;
    }
    for (size_t j = b * BlockSize; j < j_end; ++j) {
      int32_t q = static_cast<int32_t>(UnpackNibble(PackedRow, j));
      q = (q ^ 8) - 8;
      Output[j] = static_cast<float>(q - zp) * scale;
    }
  }
}

void
MLASCALL
MlasGatherBlockDequantizeSymFp16Kernel(
    uint16_t* Output,
    const uint8_t* PackedRow,
    const float* Scales,
    const uint8_t* PackedZeroPoints,
    size_t ZeroPointBase,
    size_t K,
    size_t BlockSize) {
  const size_t num_blocks = (K + BlockSize - 1) / BlockSize;
  for (size_t b = 0; b < num_blocks; ++b) {
    const float scale = Scales[b];
    int32_t zp = 0;
    if (PackedZeroPoints != nullptr) {
      zp = static_cast<int32_t>(UnpackNibble(PackedZeroPoints, ZeroPointBase + b));
      zp = (zp ^ 8) - 8;
    }
    size_t j_end = (b + 1) * BlockSize;
    if (j_end > K) {
      j_end = K;
    }
    for (size_t j = b * BlockSize; j < j_end; ++j) {
      int32_t q = static_cast<int32_t>(UnpackNibble(PackedRow, j));
      q = (q ^ 8) - 8;
      Output[j] = onnxruntime::MLFloat16(static_cast<float>(q - zp) * scale).val;
    }
  }
}

void
MLASCALL
MlasGatherBlockDequantizeAsymKernel(
    float* Output,
    const uint8_t* PackedRow,
    const float* Scales,
    const uint8_t* PackedZeroPoints,
    size_t ZeroPointBase,
    int32_t DefaultZeroPoint,
    size_t QuantBits,
    size_t K,
    size_t BlockSize) {
  const size_t num_blocks = (K + BlockSize - 1) / BlockSize;
  for (size_t b = 0; b < num_blocks; ++b) {
    const float scale = Scales[b];
    int32_t zp = DefaultZeroPoint;
    if (PackedZeroPoints != nullptr) {
      zp = UnpackZpValue(PackedZeroPoints, ZeroPointBase + b, QuantBits);
    }
    size_t j_end = (b + 1) * BlockSize;
    if (j_end > K) {
      j_end = K;
    }
    for (size_t j = b * BlockSize; j < j_end; ++j) {
      const int32_t q = static_cast<int32_t>(UnpackValue(PackedRow, j, QuantBits));
      Output[j] = static_cast<float>(q - zp) * scale;
    }
  }
}

void
MLASCALL
MlasGatherBlockDequantizeAsymFp16Kernel(
    uint16_t* Output,
    const uint8_t* PackedRow,
    const float* Scales,
    const uint8_t* PackedZeroPoints,
    size_t ZeroPointBase,
    int32_t DefaultZeroPoint,
    size_t QuantBits,
    size_t K,
    size_t BlockSize) {
  const size_t num_blocks = (K + BlockSize - 1) / BlockSize;
  for (size_t b = 0; b < num_blocks; ++b) {
    const float scale = Scales[b];
    int32_t zp = DefaultZeroPoint;
    if (PackedZeroPoints != nullptr) {
      zp = UnpackZpValue(PackedZeroPoints, ZeroPointBase + b, QuantBits);
    }
    size_t j_end = (b + 1) * BlockSize;
    if (j_end > K) {
      j_end = K;
    }
    for (size_t j = b * BlockSize; j < j_end; ++j) {
      const int32_t q = static_cast<int32_t>(UnpackValue(PackedRow, j, QuantBits));
      Output[j] = onnxruntime::MLFloat16(static_cast<float>(q - zp) * scale).val;
    }
  }
}
