/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_gather_block_dq.cpp

Abstract:

    Unit tests for the MLAS gather + block-dequantize fast-path entries:
    randomized tables vs an independent scalar oracle, bitwise float /
    1-ULP fp16. Dispatch entries run everywhere; per-ISA Avx2 entries are
    gated on the platform flag and Avx512F entries on the dispatch slot
    resolving to the AVX512F kernel (same gate idiom as MlasErfTest).

--*/

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace onnxruntime {
namespace test {

namespace {

// Deterministic PRNG (mulberry32-style, fixed seeds only).
class DeterministicRng {
 public:
  explicit DeterministicRng(uint32_t seed) : state_(seed) {}
  uint32_t Next() {
    state_ += 0x6D2B79F5;
    uint32_t t = state_;
    t = (t ^ (t >> 15)) * (t | 1);
    t ^= t + ((t ^ (t >> 7)) * (t | 61));
    return t ^ (t >> 14);
  }

 private:
  uint32_t state_;
};

// Independent scalar oracle in stored-domain math. quant_bits is 4 for
// Sym; 2, 4, or 8 for Asym.
void
OracleRow(bool sym, const uint8_t* packed, const float* scales, const uint8_t* packed_zp,
          size_t zp_base, size_t k, size_t block_size, int32_t zp_fallback, size_t quant_bits,
          float* out) {
  const size_t num_blocks = (k + block_size - 1) / block_size;
  auto unpack = [quant_bits](const uint8_t* p, size_t idx) -> int32_t {
    if (quant_bits == 8) {
      return static_cast<int32_t>(p[idx]);
    }
    const size_t shift_div = quant_bits == 4 ? 1 : 2;
    const size_t mask = quant_bits == 4 ? 1 : 3;
    return static_cast<int32_t>((p[idx >> shift_div] >> ((idx & mask) * quant_bits)) &
                                ((1u << quant_bits) - 1));
  };
  for (size_t b = 0; b < num_blocks; ++b) {
    const float scale = scales[b];
    int32_t zp = zp_fallback;
    if (packed_zp != nullptr) {
      zp = unpack(packed_zp, zp_base + b);
      if (sym) {
        zp = (zp ^ 8) - 8;
      }
    }
    size_t j_end = (b + 1) * block_size;
    if (j_end > k) {
      j_end = k;
    }
    for (size_t j = b * block_size; j < j_end; ++j) {
      int32_t q = unpack(packed, j);
      if (sym) {
        q = (q ^ 8) - 8;
      }
      out[j] = static_cast<float>(q - zp) * scale;
    }
  }
}

bool
Fp16WithinOneUlp(uint16_t a, uint16_t b) {
  if (a == b) {
    return true;
  }
  // +0.0 vs -0.0 compare equal as floats.
  if ((a == 0x0000 && b == 0x8000) || (a == 0x8000 && b == 0x0000)) {
    return true;
  }
  const uint16_t diff = a > b ? static_cast<uint16_t>(a - b) : static_cast<uint16_t>(b - a);
  return diff <= 1;
}

void
RunGatherBlockDqCase(bool sym, bool fp16, bool with_zp, size_t k, size_t block_size, size_t zp_base,
                     uint32_t seed, bool use_avx2_entry, int32_t default_zp = 8, size_t quant_bits = 4,
                     bool use_avx512f_entry = false) {
  // The AVX512F tier is FP32-output only (FP16 rows stay on AVX2); those
  // cases are covered by the Dispatch and Avx2Direct tests.
  if (use_avx512f_entry && fp16) {
    return;
  }
  DeterministicRng rng(seed);
  const size_t num_blocks = block_size == 0 ? 0 : (k + block_size - 1) / block_size;

  std::vector<uint8_t> packed((k * quant_bits + 7) / 8);
  for (size_t i = 0; i < packed.size(); ++i) {
    packed[i] = static_cast<uint8_t>(rng.Next());
  }
  const float kScales[] = {0.125f, -2.5f, 0.0f, 1.0f, 0.75f, 3.25f};
  std::vector<float> scales(num_blocks);
  for (size_t b = 0; b < num_blocks; ++b) {
    scales[b] = kScales[(b + seed) % 6];
  }
  std::vector<uint8_t> packed_zp;
  if (with_zp) {
    packed_zp.resize((zp_base + num_blocks) * quant_bits / 8 + 1);
    for (size_t i = 0; i < packed_zp.size(); ++i) {
      packed_zp[i] = static_cast<uint8_t>(rng.Next());
    }
  }
  const uint8_t* zp_ptr = with_zp ? packed_zp.data() : nullptr;
  // Null zp means the packing default: 0 for Sym, caller-picked for Asym
  // (1 << (bits-1) for uint8: 2/8/128; 0 for packed UInt4x2).
  const int32_t zp_fallback = sym ? 0 : default_zp;

  std::vector<float> expected(k);
  OracleRow(sym, packed.data(), scales.data(), zp_ptr, zp_base, k, block_size, zp_fallback,
            quant_bits, expected.data());

  MLAS_PLATFORM& platform = GetMlasPlatform();
  if (!fp16) {
    std::vector<float> actual(k, 12345.0f);
    if (use_avx512f_entry) {
// Per-ISA entries are extern "C" free functions (declared in mlasi.h under
// MLAS_TARGET_AMD64), not platform members, so call them unqualified.
#if defined(MLAS_TARGET_AMD64)
      if (sym) {
        MlasGatherBlockDequantizeSymKernelAvx512F(actual.data(), packed.data(), scales.data(),
                                                  zp_ptr, zp_base, k, block_size);
      } else {
        MlasGatherBlockDequantizeAsymKernelAvx512F(actual.data(), packed.data(), scales.data(),
                                                   zp_ptr, zp_base, default_zp, quant_bits, k,
                                                   block_size);
      }
#else
      GTEST_SKIP() << "AVX512F entries need an AMD64 build";
#endif
    } else if (use_avx2_entry) {
// Per-ISA entries are extern "C" free functions (declared in mlasi.h under
// MLAS_TARGET_AMD64), not platform members, so call them unqualified.
#if defined(MLAS_TARGET_AMD64)
      if (sym) {
        MlasGatherBlockDequantizeSymKernelAvx2(actual.data(), packed.data(), scales.data(),
                                               zp_ptr, zp_base, k, block_size);
      } else {
        MlasGatherBlockDequantizeAsymKernelAvx2(actual.data(), packed.data(), scales.data(),
                                                zp_ptr, zp_base, default_zp, quant_bits, k,
                                                block_size);
      }
#else
      GTEST_SKIP() << "AVX2 entries need an AMD64 build";
#endif
    } else {
      if (sym) {
        platform.GatherBlockDequantizeSymKernel(actual.data(), packed.data(), scales.data(), zp_ptr,
                                                zp_base, k, block_size);
      } else {
        platform.GatherBlockDequantizeAsymKernel(actual.data(), packed.data(), scales.data(), zp_ptr,
                                                 zp_base, default_zp, quant_bits, k, block_size);
      }
    }
    for (size_t j = 0; j < k; ++j) {
      EXPECT_EQ(actual[j], expected[j]) << "float mismatch at j=" << j << " k=" << k;
    }
    return;
  }

  std::vector<uint16_t> actual(k, 0xDEAD);
  // No AVX512F FP16 entries exist (FP16 rows stay on AVX2); the early return
  // above keeps use_avx512f_entry out of this half.
  if (use_avx2_entry) {
#if defined(MLAS_TARGET_AMD64)
    if (sym) {
      MlasGatherBlockDequantizeSymFp16KernelAvx2(actual.data(), packed.data(), scales.data(),
                                                 zp_ptr, zp_base, k, block_size);
    } else {
      MlasGatherBlockDequantizeAsymFp16KernelAvx2(actual.data(), packed.data(), scales.data(),
                                                  zp_ptr, zp_base, default_zp, quant_bits, k,
                                                  block_size);
    }
#else
    GTEST_SKIP() << "AVX2 entries need an AMD64 build";
#endif
  } else {
    if (sym) {
      platform.GatherBlockDequantizeSymFp16Kernel(actual.data(), packed.data(), scales.data(), zp_ptr,
                                                  zp_base, k, block_size);
    } else {
      platform.GatherBlockDequantizeAsymFp16Kernel(actual.data(), packed.data(), scales.data(), zp_ptr,
                                                   zp_base, default_zp, quant_bits, k, block_size);
    }
  }
  for (size_t j = 0; j < k; ++j) {
    const uint16_t want = onnxruntime::MLFloat16(expected[j]).val;
    EXPECT_TRUE(Fp16WithinOneUlp(actual[j], want))
        << "fp16 mismatch at j=" << j << " k=" << k << " got=" << actual[j] << " want=" << want;
  }
}

void
RunGatherBlockDqMatrix(bool use_avx2_entry, bool use_avx512f_entry = false) {
  // (sym, fp16, with_zp, K, block, zp_base, seed, ..., default_zp, bits)
  // The trailing default_zp/bits are spelled out (not defaulted) so the
  // use_avx512f_entry flag binds to its own parameter.
  RunGatherBlockDqCase(true, false, false, 32, 32, 0, 1, use_avx2_entry, 8, 4, use_avx512f_entry);
  RunGatherBlockDqCase(true, false, false, 768, 32, 0, 2, use_avx2_entry, 8, 4, use_avx512f_entry);
  RunGatherBlockDqCase(true, false, true, 768, 32, 0, 3, use_avx2_entry, 8, 4, use_avx512f_entry);
  RunGatherBlockDqCase(true, false, true, 128, 32, 5, 4, use_avx2_entry, 8, 4, use_avx512f_entry);  // odd zp base
  RunGatherBlockDqCase(false, false, true, 768, 32, 0, 5, use_avx2_entry, 8, 4, use_avx512f_entry);
  RunGatherBlockDqCase(false, false, false, 768, 32, 0, 6, use_avx2_entry, 8, 4, use_avx512f_entry);
  // Null zp with default 0: packed UInt4x2 without a zp tensor.
  RunGatherBlockDqCase(false, false, false, 768, 32, 0, 19, use_avx2_entry, 0, 4, use_avx512f_entry);
  RunGatherBlockDqCase(false, false, false, 64, 32, 0, 20, use_avx2_entry, 0, 4, use_avx512f_entry);
  RunGatherBlockDqCase(false, true, false, 64, 32, 0, 21, use_avx2_entry, 0, 4, use_avx512f_entry);
  RunGatherBlockDqCase(true, false, false, 100, 32, 0, 7, use_avx2_entry, 8, 4, use_avx512f_entry);  // K tail
  RunGatherBlockDqCase(true, false, false, 33, 16, 0, 8, use_avx2_entry, 8, 4, use_avx512f_entry);   // odd K
  RunGatherBlockDqCase(true, false, false, 0, 32, 0, 9, use_avx2_entry, 8, 4, use_avx512f_entry);    // empty row
  RunGatherBlockDqCase(true, false, false, 128, 64, 0, 10, use_avx2_entry, 8, 4, use_avx512f_entry);
  RunGatherBlockDqCase(true, false, false, 128, 128, 0, 11, use_avx2_entry, 8, 4, use_avx512f_entry);
  RunGatherBlockDqCase(true, false, false, 32, 128, 0, 12, use_avx2_entry, 8, 4, use_avx512f_entry);  // block > K
  RunGatherBlockDqCase(true, false, false, 64, 16, 0, 13, use_avx2_entry, 8, 4, use_avx512f_entry);
  RunGatherBlockDqCase(false, false, true, 256, 64, 3, 14, use_avx2_entry, 8, 4, use_avx512f_entry);
  RunGatherBlockDqCase(true, true, false, 64, 32, 0, 15, use_avx2_entry, 8, 4, use_avx512f_entry);
  RunGatherBlockDqCase(true, true, true, 128, 32, 1, 16, use_avx2_entry, 8, 4, use_avx512f_entry);
  RunGatherBlockDqCase(false, true, true, 256, 32, 0, 17, use_avx2_entry, 8, 4, use_avx512f_entry);
  RunGatherBlockDqCase(false, true, false, 33, 16, 0, 18, use_avx2_entry, 8, 4, use_avx512f_entry);
  // 2-bit Asym (uint8 defaults to zp 2): K must be a multiple of 4.
  RunGatherBlockDqCase(false, false, false, 768, 32, 0, 22, use_avx2_entry, 2, 2, use_avx512f_entry);
  RunGatherBlockDqCase(false, false, true, 768, 32, 0, 23, use_avx2_entry, 2, 2, use_avx512f_entry);
  RunGatherBlockDqCase(false, false, true, 256, 64, 3, 24, use_avx2_entry, 2, 2, use_avx512f_entry);  // odd zp base
  RunGatherBlockDqCase(false, false, false, 64, 16, 0, 25, use_avx2_entry, 2, 2, use_avx512f_entry);
  RunGatherBlockDqCase(false, false, true, 100, 32, 0, 26, use_avx2_entry, 2, 2, use_avx512f_entry);  // K tail
  RunGatherBlockDqCase(false, true, true, 64, 32, 0, 27, use_avx2_entry, 2, 2, use_avx512f_entry);
  RunGatherBlockDqCase(false, true, false, 64, 16, 0, 28, use_avx2_entry, 2, 2, use_avx512f_entry);
  // 8-bit Asym (uint8 defaults to zp 128): any K, zp unpacked per block.
  RunGatherBlockDqCase(false, false, false, 768, 32, 0, 29, use_avx2_entry, 128, 8, use_avx512f_entry);
  RunGatherBlockDqCase(false, false, true, 768, 32, 0, 30, use_avx2_entry, 128, 8, use_avx512f_entry);
  RunGatherBlockDqCase(false, false, true, 100, 32, 0, 31, use_avx2_entry, 128, 8, use_avx512f_entry);  // K tail
  RunGatherBlockDqCase(false, false, true, 33, 16, 0, 32, use_avx2_entry, 128, 8, use_avx512f_entry);   // odd K
  RunGatherBlockDqCase(false, false, false, 0, 32, 0, 33, use_avx2_entry, 128, 8, use_avx512f_entry);   // empty row
  RunGatherBlockDqCase(false, false, false, 128, 64, 0, 34, use_avx2_entry, 128, 8, use_avx512f_entry);
  RunGatherBlockDqCase(false, false, true, 256, 128, 0, 37, use_avx2_entry, 128, 8, use_avx512f_entry);  // blk128: AVX512F delegates to AVX2
  RunGatherBlockDqCase(false, true, true, 64, 32, 0, 35, use_avx2_entry, 128, 8, use_avx512f_entry);
  RunGatherBlockDqCase(false, true, false, 33, 16, 0, 36, use_avx2_entry, 128, 8, use_avx512f_entry);  // odd K
}

}  // namespace

TEST(MlasGatherBlockDqTest, Dispatch) {
  // Runtime dispatch selects the best kernel for the host; every matrix
  // case must match the oracle through it.
  RunGatherBlockDqMatrix(/*use_avx2_entry=*/false);
}

TEST(MlasGatherBlockDqTest, Avx2Direct) {
  if (!GetMlasPlatform().Avx2Supported_) {
    GTEST_SKIP() << "AVX2 not available on this host";
  }
  RunGatherBlockDqMatrix(/*use_avx2_entry=*/true);
}

TEST(MlasGatherBlockDqTest, Avx512FDirect) {
#if defined(MLAS_TARGET_AMD64)
  // Same gate idiom as MlasErfTest::Avx512Available: run only where runtime
  // dispatch selected the AVX512F kernel for this host. FP32 cases only (the
  // tier has no FP16 entries); FP16 coverage comes from Dispatch/Avx2Direct.
  if (GetMlasPlatform().GatherBlockDequantizeSymKernel != MlasGatherBlockDequantizeSymKernelAvx512F) {
    GTEST_SKIP() << "AVX512F dispatch not selected on this host";
  }
#else
  GTEST_SKIP() << "AVX512F entries need an AMD64 build";
#endif
  RunGatherBlockDqMatrix(/*use_avx2_entry=*/false, /*use_avx512f_entry=*/true);
}

}  // namespace test
}  // namespace onnxruntime
