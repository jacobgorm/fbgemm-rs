use crate::kernels::{get_kernels, GemmParams};
use crate::pack::{PackedMatrix, BLOCK_COL_SIZE};
use crate::partition::PARTITION_AVX2;

const MB_MAX: usize = 120;

/// Transpose A from row-major to column-major scratchpad (used on x86_64).
/// to[r + c * nrow] = from[r * ldim + c]
#[cfg(not(target_arch = "aarch64"))]
fn pack_a(nrow: usize, ncol: usize, from: &[f32], ldim: usize, to: &mut [f32]) {
    for r in 0..nrow {
        for c in 0..ncol {
            to[r + c * nrow] = from[r * ldim + c];
        }
    }
}

/// Compute C = beta * C + A * packed_B (single-threaded).
///
/// - `m`: number of rows of A and C
/// - `a`: M×K row-major matrix (length m*k)
/// - `packed_b`: pre-packed K×N matrix
/// - `beta`: scaling factor for existing C
/// - `c`: M×N row-major output matrix (length m*n)
pub fn cblas_gemm_compute(
    m: usize,
    a: &[f32],
    packed_b: &PackedMatrix,
    beta: f32,
    c: &mut [f32],
) {
    let k = packed_b.k();
    let n = packed_b.n();
    let ldc = n;
    let bcol = BLOCK_COL_SIZE;
    let brow = packed_b.block_row_size();
    let kernels = get_kernels();

    #[cfg(not(target_arch = "aarch64"))]
    let mut scratchpad = vec![0.0f32; 256 * 1024];

    let mut gp = GemmParams {
        k: 0,
        a: std::ptr::null_mut(),
        b: std::ptr::null(),
        beta: 0.0,
        _pad: 0,
        c: std::ptr::null_mut(),
        ldc: 0,
        b_block_cols: 0,
        lda: 0,
    };

    for m0 in (0..m).step_by(MB_MAX) {
        let mb = MB_MAX.min(m - m0);
        assert!(mb <= 120);

        for k_ind in (0..k).step_by(brow) {
            let beta_ = if k_ind == 0 { beta } else { 1.0 };
            let kb = brow.min(k - k_ind);

            let mut m1 = m0;
            let partition = &PARTITION_AVX2[mb];
            for cycle in partition {
                let kernel_nrows = cycle[0] as usize;
                let nkernel_nrows = cycle[1] as usize;
                if kernel_nrows == 0 {
                    break;
                }

                for _ in 0..nkernel_nrows {
                    let m2 = m1;
                    m1 += kernel_nrows;

                    // Set up A pointer
                    #[cfg(target_arch = "aarch64")]
                    {
                        gp.a = unsafe { (a.as_ptr() as *mut f32).add(m2 * k + k_ind) };
                        gp.lda = (k * 4) as u64;
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        if m == 1 {
                            gp.a = unsafe { (a.as_ptr() as *mut f32).add(k_ind) };
                        } else {
                            pack_a(
                                kernel_nrows,
                                kb,
                                &a[m2 * k + k_ind..],
                                k,
                                &mut scratchpad,
                            );
                            gp.a = scratchpad.as_mut_ptr();
                        }
                        gp.lda = 0; // not used on x86_64 (A is transposed)
                    }

                    let nbcol = n / bcol;
                    gp.k = kb as u64;
                    gp.b = packed_b.at(k_ind, 0);
                    gp.beta = beta_;
                    gp.c = unsafe { (c.as_ptr() as *mut f32).add(m2 * ldc) };
                    gp.ldc = (ldc * 4) as u64;
                    gp.b_block_cols = nbcol as u64;

                    let kernel = kernels[kernel_nrows]
                        .expect("no kernel for this nrows");

                    if n % bcol == 0 {
                        if nbcol > 0 {
                            unsafe { kernel(&mut gp) };
                        }
                    } else {
                        // Handle aligned blocks
                        if nbcol > 0 {
                            unsafe { kernel(&mut gp) };
                        }

                        // Handle fringe (remaining columns)
                        let last_blk_col = nbcol * bcol;
                        let rem = n - last_blk_col;
                        debug_assert!(rem < bcol);

                        let mut c_tmp = [0.0f32; 14 * 32];
                        gp.b = packed_b.at(k_ind, last_blk_col);
                        gp.c = c_tmp.as_mut_ptr();
                        gp.ldc = (bcol * 4) as u64;
                        gp.b_block_cols = 1;
                        unsafe { kernel(&mut gp) };

                        // Copy valid columns back to C
                        for i in 0..kernel_nrows {
                            for j in 0..rem {
                                let src = c_tmp[i * bcol + j];
                                let dst = &mut c[(m2 + i) * ldc + last_blk_col + j];
                                if beta_ == 0.0 {
                                    *dst = src;
                                } else {
                                    *dst = beta_ * *dst + src;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
