use super::pack::*;

#[cfg(target_arch = "x86_64")]
use super::avx2;
#[cfg(target_arch = "x86_64")]
use super::avx512_vnni;

#[cfg(target_arch = "aarch64")]
use super::neon;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Reusable scratch buffers for quantized GEMM.
/// Create once and pass to [`i8gemm_compute_with_scratch`] /
/// [`i8gemm_compute_par_with_scratch`] to avoid per-call allocation.
pub struct I8GemmScratch {
    a_u8: Vec<u8>,
    a_scales: Vec<f32>,
    a_zero_points: Vec<i32>,
    c_i32: Vec<i32>,
}

impl I8GemmScratch {
    pub fn new() -> Self {
        Self {
            a_u8: Vec::new(),
            a_scales: Vec::new(),
            a_zero_points: Vec::new(),
            c_i32: Vec::new(),
        }
    }
}

/// Detect SIMD capability flags at runtime.
struct SimdFlags {
    #[cfg(target_arch = "x86_64")]
    avx512_vnni: bool,
    #[cfg(target_arch = "x86_64")]
    avx2: bool,
    #[cfg(target_arch = "aarch64")]
    neon_dotprod: bool,
}

impl SimdFlags {
    fn detect() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            avx512_vnni: is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512vnni"),
            #[cfg(target_arch = "x86_64")]
            avx2: is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "aarch64")]
            neon_dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
        }
    }

    fn use_simd(&self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            self.avx512_vnni || self.avx2
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.neon_dotprod
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Effective zero point: shifted by 128 on the Neon SDOT path.
    fn effective_zp(&self, a_zero_point: i32) -> i32 {
        #[cfg(target_arch = "aarch64")]
        if self.neon_dotprod {
            return a_zero_point - 128;
        }
        a_zero_point
    }
}

// SAFETY: SimdFlags is read-only after construction.
unsafe impl Sync for SimdFlags {}

/// Main dispatcher for quantized GEMM: float32 activations × int8 weights → float32 output.
#[cfg(not(feature = "rayon"))]
pub fn i8gemm_compute(
    m: usize,
    a_float: &[f32],
    packed_b: &PackedBMatrixI8,
    b_scale: f32,
    c_float: &mut [f32],
) {
    i8gemm_compute_with_scratch(
        m,
        a_float,
        packed_b,
        b_scale,
        c_float,
        &mut I8GemmScratch::new(),
    );
}

/// Like [`i8gemm_compute`] but reuses caller-provided scratch buffers.
#[cfg(not(feature = "rayon"))]
pub fn i8gemm_compute_with_scratch(
    m: usize,
    a_float: &[f32],
    packed_b: &PackedBMatrixI8,
    b_scale: f32,
    c_float: &mut [f32],
    scratch: &mut I8GemmScratch,
) {
    let k = packed_b.k();
    let n = packed_b.n();

    quantize_a_per_row_into_no_offsets(
        a_float,
        m,
        k,
        &mut scratch.a_u8,
        &mut scratch.a_scales,
        &mut scratch.a_zero_points,
    );
    scratch.c_i32.resize(m * n, 0);
    scratch.c_i32.fill(0);
    let flags = SimdFlags::detect();

    for mb_start in (0..m).step_by(MCB) {
        let mc = std::cmp::min(MCB, m - mb_start);
        process_m_block(
            mb_start,
            mc,
            &scratch.a_u8,
            k,
            n,
            packed_b,
            &mut scratch.c_i32,
            &flags,
        );
    }

    dequantize_per_row(
        m,
        n,
        &scratch.c_i32,
        c_float,
        &scratch.a_scales,
        b_scale,
        &scratch.a_zero_points,
        packed_b.col_offsets(),
        &flags,
    );
}

/// Parallel version of [`i8gemm_compute`] using rayon.
///
/// M-blocks are dispatched across threads. Each writes to disjoint rows of the
/// accumulation buffer, so no synchronization is needed.
#[cfg(feature = "rayon")]
pub fn i8gemm_compute_par(
    m: usize,
    a_float: &[f32],
    packed_b: &PackedBMatrixI8,
    b_scale: f32,
    c_float: &mut [f32],
) {
    i8gemm_compute_par_with_scratch(
        m,
        a_float,
        packed_b,
        b_scale,
        c_float,
        &mut I8GemmScratch::new(),
    );
}

/// Like [`i8gemm_compute_par`] but reuses caller-provided scratch buffers.
#[cfg(feature = "rayon")]
pub fn i8gemm_compute_par_with_scratch(
    m: usize,
    a_float: &[f32],
    packed_b: &PackedBMatrixI8,
    b_scale: f32,
    c_float: &mut [f32],
    scratch: &mut I8GemmScratch,
) {
    let k = packed_b.k();
    let n = packed_b.n();

    quantize_a_per_row_into_no_offsets(
        a_float,
        m,
        k,
        &mut scratch.a_u8,
        &mut scratch.a_scales,
        &mut scratch.a_zero_points,
    );
    scratch.c_i32.resize(m * n, 0);
    scratch.c_i32.fill(0);
    let flags = SimdFlags::detect();

    let mb_starts: Vec<usize> = (0..m).step_by(MCB).collect();
    let c_ptr = scratch.c_i32.as_mut_ptr() as usize;

    let num_k_blocks = packed_b.num_k_blocks();
    let num_n_blocks = packed_b.num_n_blocks();
    let split_n_blocks = flags.use_simd() && n % NR == 0 && num_n_blocks > 1;
    let n_chunks = if split_n_blocks {
        rayon::current_num_threads().min(num_n_blocks)
    } else {
        1
    };
    let n_chunk_blocks = num_n_blocks.div_ceil(n_chunks);

    for kb in 0..num_k_blocks {
        let k_start = kb * KCB;
        let kc = std::cmp::min(KCB, k - k_start);
        let tasks: Vec<_> = mb_starts
            .iter()
            .flat_map(|&mb_start| {
                (0..num_n_blocks)
                    .step_by(n_chunk_blocks)
                    .map(move |nb_start| {
                        let nb_end = (nb_start + n_chunk_blocks).min(num_n_blocks);
                        (mb_start, nb_start, nb_end)
                    })
            })
            .collect();

        tasks.par_iter().for_each(|&(mb_start, nb_start, nb_end)| {
            let mc = std::cmp::min(MCB, m - mb_start);
            let c_ptr = c_ptr as *mut i32;
            // SAFETY: tasks write disjoint row/column tiles of C.
            unsafe {
                process_kb_block(
                    mb_start,
                    mc,
                    kb,
                    k_start,
                    kc,
                    &scratch.a_u8,
                    k,
                    n,
                    nb_start,
                    nb_end,
                    packed_b,
                    c_ptr,
                    &flags,
                );
            }
        });
    }

    dequantize_per_row(
        m,
        n,
        &scratch.c_i32,
        c_float,
        &scratch.a_scales,
        b_scale,
        &scratch.a_zero_points,
        packed_b.col_offsets(),
        &flags,
    );
}

/// Process one M-block across all K-blocks and N-blocks (sequential path).
#[cfg(not(feature = "rayon"))]
fn process_m_block(
    mb_start: usize,
    mc: usize,
    a_u8: &[u8],
    k: usize,
    n: usize,
    packed_b: &PackedBMatrixI8,
    c_i32: &mut [i32],
    flags: &SimdFlags,
) {
    let num_k_blocks = packed_b.num_k_blocks();
    let num_n_blocks = packed_b.num_n_blocks();

    for kb in 0..num_k_blocks {
        let k_start = kb * KCB;
        let kc = std::cmp::min(KCB, k - k_start);
        // SAFETY: single-threaded, exclusive access to c_i32.
        unsafe {
            process_kb_block(
                mb_start,
                mc,
                kb,
                k_start,
                kc,
                a_u8,
                k,
                n,
                0,
                num_n_blocks,
                packed_b,
                c_i32.as_mut_ptr(),
                flags,
            );
        }
    }
}

/// Process one (M-block, K-block) pair across all N-blocks.
///
/// # Safety
/// Caller must ensure rows `[mb_start..mb_start+mc]` of the C buffer pointed
/// to by `c_ptr` are not concurrently modified.
unsafe fn process_kb_block(
    mb_start: usize,
    mc: usize,
    kb: usize,
    k_start: usize,
    kc: usize,
    a_u8: &[u8],
    k: usize,
    n: usize,
    nb_start: usize,
    nb_end: usize,
    packed_b: &PackedBMatrixI8,
    c_ptr: *mut i32,
    flags: &SimdFlags,
) {
    let kc_aligned = (kc + ROW_INTERLEAVE - 1) / ROW_INTERLEAVE * ROW_INTERLEAVE;

    // Thread-local scratchpad for A repacking
    let mut a_packed = if flags.use_simd() {
        vec![0u8; MR * KCB]
    } else {
        Vec::new()
    };

    for nb in nb_start..nb_end {
        let n_start = nb * NR;
        let nc = std::cmp::min(NR, n - n_start);
        let b_tile = packed_b.tile(kb, nb);

        let mut row_offset = 0;
        while row_offset < mc {
            let kernel_rows = std::cmp::min(MR, mc - row_offset);
            let i = mb_start + row_offset;

            #[cfg(target_arch = "x86_64")]
            if flags.avx512_vnni && nc == NR {
                a_packed.fill(0);
                avx2::pack_a_tile(a_u8, i, k_start, kernel_rows, kc, k, &mut a_packed);
                avx512_vnni::dispatch_i8_kernel(
                    kernel_rows,
                    a_packed.as_ptr(),
                    b_tile.as_ptr(),
                    c_ptr.add(i * n + n_start),
                    kc_aligned,
                    n * 4,
                );
                row_offset += kernel_rows;
                continue;
            }

            #[cfg(target_arch = "x86_64")]
            if flags.avx2 && nc == NR {
                a_packed.fill(0);
                avx2::pack_a_tile(a_u8, i, k_start, kernel_rows, kc, k, &mut a_packed);
                avx2::dispatch_i8_kernel(
                    kernel_rows,
                    a_packed.as_ptr(),
                    b_tile.as_ptr(),
                    c_ptr.add(i * n + n_start),
                    kc_aligned,
                    n * 4,
                );
                row_offset += kernel_rows;
                continue;
            }

            #[cfg(target_arch = "aarch64")]
            if flags.neon_dotprod && nc == NR {
                a_packed.fill(0);
                neon::pack_a_tile(a_u8, i, k_start, kernel_rows, kc, k, &mut a_packed);
                neon::dispatch_i8_kernel(
                    kernel_rows,
                    a_packed.as_ptr(),
                    b_tile.as_ptr(),
                    c_ptr.add(i * n + n_start),
                    kc_aligned,
                    n * 4,
                );
                row_offset += kernel_rows;
                continue;
            }

            // Fallback: reference kernel
            let c_slice = std::slice::from_raw_parts_mut(
                c_ptr.add(i * n + n_start),
                (kernel_rows - 1) * n + nc,
            );

            #[cfg(target_arch = "aarch64")]
            if flags.neon_dotprod {
                // Signed A to match SDOT's int8×int8 semantics
                ref_i8i8_kernel(
                    kernel_rows,
                    &a_u8[i * k + k_start..],
                    k,
                    b_tile,
                    c_slice,
                    n,
                    kc,
                    nc,
                );
                row_offset += kernel_rows;
                continue;
            }

            ref_u8i8_kernel(
                kernel_rows,
                &a_u8[i * k + k_start..],
                k,
                b_tile,
                c_slice,
                n,
                kc,
                nc,
            );

            row_offset += kernel_rows;
        }
    }
}

/// Dequantize int32 accumulation buffer to float32 output.
fn dequantize_per_row(
    m: usize,
    n: usize,
    c_i32: &[i32],
    c_float: &mut [f32],
    a_scales: &[f32],
    b_scale: f32,
    a_zero_points: &[i32],
    col_offsets: &[i32],
    flags: &SimdFlags,
) {
    debug_assert_eq!(a_scales.len(), m);
    debug_assert_eq!(a_zero_points.len(), m);
    for i in 0..m {
        let output_scale = a_scales[i] * b_scale;
        let effective_zp = flags.effective_zp(a_zero_points[i]);
        for j in 0..n {
            let raw = c_i32[i * n + j];
            let adjusted = raw - effective_zp * col_offsets[j];
            c_float[i * n + j] = adjusted as f32 * output_scale;
        }
    }
}

/// Reference kernel: uint8 A × int8 B → int32 C.
fn ref_u8i8_kernel(
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

/// Reference kernel: int8 A × int8 B → int32 C (Neon SDOT path column fringe).
#[cfg(target_arch = "aarch64")]
fn ref_i8i8_kernel(
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
            let a_val = (a[i * lda + k_idx] ^ 0x80) as i8 as i32;
            for j in 0..nc {
                let b_val = b[g * NR * ROW_INTERLEAVE + j * ROW_INTERLEAVE + ri] as i32;
                c[i * ldc + j] += a_val * b_val;
            }
        }
    }
}
