use super::pack::*;

#[cfg(target_arch = "x86_64")]
use super::avx2;

/// Main dispatcher for quantized GEMM: float32 activations × int8 weights → float32 output.
pub fn i8gemm_compute(
    m: usize,
    a_float: &[f32],
    packed_b: &PackedBMatrixI8,
    b_scale: f32,
    c_float: &mut [f32],
) {
    let k = packed_b.k();
    let n = packed_b.n();

    // Step 1: Quantize activations to uint8
    let (a_u8, a_scale, a_zero_point, _row_offsets) = quantize_a(a_float, m, k);

    // Step 2: Allocate int32 accumulation buffer
    let mut c_i32 = vec![0i32; m * n];

    // Step 3: Blocked GEMM
    let num_k_blocks = packed_b.num_k_blocks();
    let num_n_blocks = packed_b.num_n_blocks();

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    // Scratchpad for A repacking (KCB-strided) when using SIMD kernels
    #[cfg(target_arch = "x86_64")]
    let mut a_packed = if use_avx2 {
        vec![0u8; MR * KCB]
    } else {
        Vec::new()
    };

    for mb_start in (0..m).step_by(MCB) {
        let mc = std::cmp::min(MCB, m - mb_start);

        for kb in 0..num_k_blocks {
            let k_start = kb * KCB;
            let kc = std::cmp::min(KCB, k - k_start);

            for nb in 0..num_n_blocks {
                let n_start = nb * NR;
                let nc = std::cmp::min(NR, n - n_start);
                let b_tile = packed_b.tile(kb, nb);

                let mut row_offset = 0;
                while row_offset < mc {
                    let kernel_rows = std::cmp::min(MR, mc - row_offset);
                    let i = mb_start + row_offset;

                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 && kernel_rows == MR && nc == NR {
                        let kc_aligned = (kc + ROW_INTERLEAVE - 1) / ROW_INTERLEAVE * ROW_INTERLEAVE;
                        // Pack A tile with KCB stride for SIMD kernel
                        a_packed.fill(0);
                        avx2::pack_a_tile(
                            &a_u8, i, k_start, MR, kc, k, &mut a_packed,
                        );
                        unsafe {
                            avx2::avx2_i8_kernel_12(
                                a_packed.as_ptr(),
                                b_tile.as_ptr(),
                                c_i32[i * n + n_start..].as_mut_ptr(),
                                kc_aligned,
                                n * 4, // ldc in bytes
                            );
                        }
                        row_offset += kernel_rows;
                        continue;
                    }

                    // Fallback: reference kernel
                    ref_i8_kernel(
                        kernel_rows,
                        &a_u8[i * k + k_start..],
                        k,
                        b_tile,
                        &mut c_i32[i * n + n_start..],
                        n,
                        kc,
                        nc,
                    );

                    row_offset += kernel_rows;
                }
            }
        }
    }

    // Step 4: Dequantize: C_f32 = (C_i32 - a_zp * col_offsets) * a_scale * b_scale
    let output_scale = a_scale * b_scale;
    let col_offsets = packed_b.col_offsets();
    for i in 0..m {
        for j in 0..n {
            let raw = c_i32[i * n + j];
            let adjusted = raw - a_zero_point * col_offsets[j];
            c_float[i * n + j] = adjusted as f32 * output_scale;
        }
    }
}

/// Reference kernel: mc rows × nc columns × kc K-elements.
/// B is in row-interleaved format within the tile.
fn ref_i8_kernel(
    mc: usize,
    a: &[u8],
    lda: usize,
    b: &[i8],
    c: &mut [i32],
    ldc: usize,
    kc: usize,
    nc: usize,
) {
    for i in 0..mc {
        for k_idx in 0..kc {
            let g = k_idx / ROW_INTERLEAVE;
            let ri = k_idx % ROW_INTERLEAVE;
            let a_val = a[i * lda + k_idx] as i32;
            for j in 0..nc {
                let b_val = b[g * NR * ROW_INTERLEAVE + j * ROW_INTERLEAVE + ri] as i32;
                c[i * ldc + j] += a_val * b_val;
            }
        }
    }
}
