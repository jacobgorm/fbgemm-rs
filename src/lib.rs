//! Rust bindings for FBGEMM's optimized SGEMM (single-precision matrix multiply).
//!
//! Computes C = beta * C + A * B where A, B, C are f32 matrices in row-major order.
//!
//! Matrix B must be pre-packed using [`PackedMatrix`] for optimal performance.
//! The packed representation can be reused across multiple GEMM calls with
//! different A matrices (e.g., different batch inputs against the same weights).

use std::ptr::NonNull;

mod ffi {
    #[repr(C)]
    pub struct FbgemmPackedMatrix {
        _opaque: [u8; 0],
    }

    unsafe extern "C" {
        pub fn fbgemm_packed_matrix_new(
            trans: i32,
            nrow: i32,
            ncol: i32,
            alpha: f32,
            data: *const f32,
        ) -> *mut FbgemmPackedMatrix;

        pub fn fbgemm_packed_matrix_free(mat: *mut FbgemmPackedMatrix);

        pub fn fbgemm_packed_matrix_nrow(mat: *const FbgemmPackedMatrix) -> i32;
        pub fn fbgemm_packed_matrix_ncol(mat: *const FbgemmPackedMatrix) -> i32;

        pub fn fbgemm_sgemm(
            m: i32,
            a: *const f32,
            packed_b: *const FbgemmPackedMatrix,
            beta: f32,
            c: *mut f32,
            num_threads: i32,
        );
    }
}

/// A pre-packed matrix B in FBGEMM's optimized blocked layout.
///
/// Create from a K×N row-major matrix. The packed form is reusable
/// for repeated GEMM calls with different A matrices.
pub struct PackedMatrix {
    ptr: NonNull<ffi::FbgemmPackedMatrix>,
}

// PackedMatrix is read-only after creation, safe to share across threads.
unsafe impl Send for PackedMatrix {}
unsafe impl Sync for PackedMatrix {}

impl PackedMatrix {
    /// Pack a K×N row-major matrix.
    ///
    /// `data` must have length `k * n`.
    pub fn new(k: usize, n: usize, data: &[f32]) -> Self {
        assert_eq!(data.len(), k * n, "data length must be k * n");
        let ptr = unsafe {
            ffi::fbgemm_packed_matrix_new(0, k as i32, n as i32, 1.0, data.as_ptr())
        };
        let ptr = NonNull::new(ptr).expect("FBGEMM PackedMatrix allocation failed");
        Self { ptr }
    }

    /// Pack a K×N matrix stored in column-major (transposed) order.
    ///
    /// `data` must have length `k * n`, stored as N×K row-major (i.e., K×N column-major).
    pub fn from_transposed(k: usize, n: usize, data: &[f32]) -> Self {
        assert_eq!(data.len(), k * n, "data length must be k * n");
        let ptr = unsafe {
            ffi::fbgemm_packed_matrix_new(1, k as i32, n as i32, 1.0, data.as_ptr())
        };
        let ptr = NonNull::new(ptr).expect("FBGEMM PackedMatrix allocation failed");
        Self { ptr }
    }

    /// Pack with a scaling factor applied: packed = alpha * B.
    pub fn with_alpha(k: usize, n: usize, data: &[f32], alpha: f32) -> Self {
        assert_eq!(data.len(), k * n, "data length must be k * n");
        let ptr = unsafe {
            ffi::fbgemm_packed_matrix_new(0, k as i32, n as i32, alpha, data.as_ptr())
        };
        let ptr = NonNull::new(ptr).expect("FBGEMM PackedMatrix allocation failed");
        Self { ptr }
    }

    pub fn k(&self) -> usize {
        unsafe { ffi::fbgemm_packed_matrix_nrow(self.ptr.as_ptr()) as usize }
    }

    pub fn n(&self) -> usize {
        unsafe { ffi::fbgemm_packed_matrix_ncol(self.ptr.as_ptr()) as usize }
    }
}

impl Drop for PackedMatrix {
    fn drop(&mut self) {
        unsafe { ffi::fbgemm_packed_matrix_free(self.ptr.as_ptr()) }
    }
}

/// Compute C = beta * C + A * B.
///
/// - `a`: M×K row-major matrix
/// - `packed_b`: pre-packed K×N matrix (see [`PackedMatrix`])
/// - `c`: M×N row-major output matrix
/// - `m`: number of rows in A and C
/// - `beta`: scaling factor for existing C values (0.0 to overwrite)
///
/// # Panics
///
/// Panics if slice lengths don't match the declared dimensions.
pub fn sgemm(m: usize, a: &[f32], packed_b: &PackedMatrix, beta: f32, c: &mut [f32]) {
    let k = packed_b.k();
    let n = packed_b.n();
    assert_eq!(a.len(), m * k, "a must have length m * k");
    assert_eq!(c.len(), m * n, "c must have length m * n");

    unsafe {
        ffi::fbgemm_sgemm(
            m as i32,
            a.as_ptr(),
            packed_b.ptr.as_ptr(),
            beta,
            c.as_mut_ptr(),
            1,
        );
    }
}

/// Compute C = A * B (overwriting C).
///
/// Convenience wrapper for `sgemm` with `beta = 0.0`.
pub fn sgemm_simple(m: usize, a: &[f32], packed_b: &PackedMatrix, c: &mut [f32]) {
    sgemm(m, a, packed_b, 0.0, c);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sgemm() {
        // A = [[1, 2, 3],
        //      [4, 5, 6]]  (2x3)
        // B = [[7, 8],
        //      [9, 10],
        //      [11, 12]]   (3x2)
        // C = A * B = [[58, 64],
        //              [139, 154]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0f32; 4];

        let packed_b = PackedMatrix::new(3, 2, &b);
        assert_eq!(packed_b.k(), 3);
        assert_eq!(packed_b.n(), 2);

        sgemm_simple(2, &a, &packed_b, &mut c);

        assert!((c[0] - 58.0).abs() < 1e-4, "c[0] = {}", c[0]);
        assert!((c[1] - 64.0).abs() < 1e-4, "c[1] = {}", c[1]);
        assert!((c[2] - 139.0).abs() < 1e-4, "c[2] = {}", c[2]);
        assert!((c[3] - 154.0).abs() < 1e-4, "c[3] = {}", c[3]);
    }

    #[test]
    fn test_sgemm_with_beta() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let b = vec![3.0, 4.0, 5.0, 6.0]; // 2x2
        let mut c = vec![1.0, 1.0, 1.0, 1.0]; // 2x2 ones

        let packed_b = PackedMatrix::new(2, 2, &b);

        // C = 2.0 * C + A * B = [[2+3, 2+4], [2+5, 2+6]] = [[5, 6], [7, 8]]
        sgemm(2, &a, &packed_b, 2.0, &mut c);

        assert!((c[0] - 5.0).abs() < 1e-4, "c[0] = {}", c[0]);
        assert!((c[1] - 6.0).abs() < 1e-4, "c[1] = {}", c[1]);
        assert!((c[2] - 7.0).abs() < 1e-4, "c[2] = {}", c[2]);
        assert!((c[3] - 8.0).abs() < 1e-4, "c[3] = {}", c[3]);
    }

    #[test]
    fn test_larger_matrix() {
        let m = 16;
        let k = 32;
        let n = 16;

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];

        let packed_b = PackedMatrix::new(k, n, &b);
        sgemm_simple(m, &a, &packed_b, &mut c);

        // Verify against naive reference
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    c_ref[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }

        for i in 0..m * n {
            assert!(
                (c[i] - c_ref[i]).abs() < 1e-2,
                "mismatch at {}: got {}, expected {}",
                i,
                c[i],
                c_ref[i]
            );
        }
    }
}
