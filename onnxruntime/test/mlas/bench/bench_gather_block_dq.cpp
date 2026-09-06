// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/mlas/lib/mlasi.h"
#include "test/mlas/bench/bench_util.h"

// Row-dequantize throughput for the gather + block-dequantize fast path:
// one K-element row (K=768, the embedding shape) through each ISA tier.
// range(0) = tier (0 dispatch, 1 scalar, 2 AVX2, 3 AVX512F), range(1) = sym,
// range(2) = fp16 out, range(3) = bits, range(4) = block size,
// range(5) = explicit zp tensor.
static void BM_GatherBlockDequantizeRow(benchmark::State& state) {
  const int tier = (int)state.range(0);
  const bool sym = state.range(1) != 0;
  const bool fp16 = state.range(2) != 0;
  const size_t bits = (size_t)state.range(3);
  const size_t block_size = (size_t)state.range(4);
  const bool with_zp = state.range(5) != 0;
  constexpr size_t kK = 768;

  if (sym && bits != 4) {
    state.SkipWithError("sym entries are 4-bit only");
    return;
  }
  MLAS_PLATFORM& platform = GetMlasPlatform();
#if !defined(MLAS_TARGET_AMD64)
  if (tier >= 2) {
    state.SkipWithError("per-ISA entries need an AMD64 build");
    return;
  }
#endif
  if (tier == 2 && !platform.Avx2Supported_) {
    state.SkipWithError("AVX2 not available on this host");
    return;
  }
  if (tier == 3) {
#if defined(MLAS_TARGET_AMD64)
    if (platform.GatherBlockDequantizeSymKernel != MlasGatherBlockDequantizeSymKernelAvx512F) {
      state.SkipWithError("AVX512F dispatch not selected on this host");
      return;
    }
    if (bits != 4 && bits != 8) {
      // No AVX512F vector path for 2-bit rows (they delegate to the AVX2
      // entry): tier 3 would duplicate tier 2 exactly, so skip it.
      state.SkipWithError("AVX512F core is 4/8-bit only (2-bit delegates to AVX2)");
      return;
    }
    if (fp16) {
      state.SkipWithError("AVX512F tier is FP32-output only (FP16 rows stay on AVX2)");
      return;
    }
#else
    state.SkipWithError("AVX512F entries need an AMD64 build");
    return;
#endif
  }

  const size_t num_blocks = (kK + block_size - 1) / block_size;
  std::vector<uint8_t> packed = RandomVectorUniform<uint8_t>(kK);  // oversized for sub-byte
                                                                  // widths; prefix used
  std::vector<float> scales = RandomVectorUniform<float>(num_blocks, -2.0f, 2.0f);
  std::vector<uint8_t> zp = RandomVectorUniform<uint8_t>(num_blocks);  // one byte per block
                                                                       // covers all widths
  const uint8_t* zp_ptr = with_zp ? zp.data() : nullptr;
  // Null-zp default matches the op: 0 for Sym, 1 << (bits-1) for Asym.
  const int32_t default_zp = sym ? 0 : (1 << (bits - 1));
  std::vector<float> out_f32(kK);
  std::vector<uint16_t> out_f16(kK);

  for (auto _ : state) {
    if (!fp16) {
      benchmark::DoNotOptimize(out_f32.data());
      float* out = out_f32.data();
      switch (tier) {
        case 1:
          if (sym) {
            MlasGatherBlockDequantizeSymKernel(out, packed.data(), scales.data(), zp_ptr, 0, kK,
                                               block_size);
          } else {
            MlasGatherBlockDequantizeAsymKernel(out, packed.data(), scales.data(), zp_ptr, 0,
                                                default_zp, bits, kK, block_size);
          }
          break;
        case 2:
#if defined(MLAS_TARGET_AMD64)
          if (sym) {
            MlasGatherBlockDequantizeSymKernelAvx2(out, packed.data(), scales.data(), zp_ptr, 0,
                                                   kK, block_size);
          } else {
            MlasGatherBlockDequantizeAsymKernelAvx2(out, packed.data(), scales.data(), zp_ptr, 0,
                                                    default_zp, bits, kK, block_size);
          }
#endif
          break;
        case 3:
#if defined(MLAS_TARGET_AMD64)
          if (sym) {
            MlasGatherBlockDequantizeSymKernelAvx512F(out, packed.data(), scales.data(), zp_ptr, 0,
                                                      kK, block_size);
          } else {
            MlasGatherBlockDequantizeAsymKernelAvx512F(out, packed.data(), scales.data(), zp_ptr, 0,
                                                       default_zp, bits, kK, block_size);
          }
#endif
          break;
        default:
          if (sym) {
            platform.GatherBlockDequantizeSymKernel(out, packed.data(), scales.data(), zp_ptr, 0,
                                                    kK, block_size);
          } else {
            platform.GatherBlockDequantizeAsymKernel(out, packed.data(), scales.data(), zp_ptr, 0,
                                                     default_zp, bits, kK, block_size);
          }
          break;
      }
      benchmark::ClobberMemory();
    } else {
      benchmark::DoNotOptimize(out_f16.data());
      uint16_t* out = out_f16.data();
      switch (tier) {
        case 1:
          if (sym) {
            MlasGatherBlockDequantizeSymFp16Kernel(out, packed.data(), scales.data(), zp_ptr, 0,
                                                   kK, block_size);
          } else {
            MlasGatherBlockDequantizeAsymFp16Kernel(out, packed.data(), scales.data(), zp_ptr, 0,
                                                    default_zp, bits, kK, block_size);
          }
          break;
        case 2:
#if defined(MLAS_TARGET_AMD64)
          if (sym) {
            MlasGatherBlockDequantizeSymFp16KernelAvx2(out, packed.data(), scales.data(), zp_ptr, 0,
                                                       kK, block_size);
          } else {
            MlasGatherBlockDequantizeAsymFp16KernelAvx2(out, packed.data(), scales.data(), zp_ptr,
                                                        0, default_zp, bits, kK, block_size);
          }
#endif
          break;
        default:
          // Tier 3 (AVX512F) has no FP16 entries; those combos skip up front,
          // so dispatch (AVX2 FP16 on this host) is only a fallback here.
          if (sym) {
            platform.GatherBlockDequantizeSymFp16Kernel(out, packed.data(), scales.data(), zp_ptr,
                                                        0, kK, block_size);
          } else {
            platform.GatherBlockDequantizeAsymFp16Kernel(out, packed.data(), scales.data(), zp_ptr,
                                                         0, default_zp, bits, kK, block_size);
          }
          break;
      }
      benchmark::ClobberMemory();
    }
  }
  state.SetItemsProcessed(state.iterations() * kK);
}

BENCHMARK(BM_GatherBlockDequantizeRow)
    ->ArgsProduct({
        {0, 1, 2, 3},  // tier
        {0, 1},        // sym
        {0, 1},        // fp16 out
        {2, 4, 8},      // bits
        {16, 32, 64, 128},  // block size
        {0, 1},         // explicit zp
    });
