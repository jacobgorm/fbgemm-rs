#ifndef FBGEMM_SGEMM_H
#define FBGEMM_SGEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to PackedGemmMatrixB<float>
typedef struct FbgemmPackedMatrix FbgemmPackedMatrix;

// Pack matrix B (K x N, row-major) into FBGEMM's internal blocked format.
// trans: 0 = NoTranspose, 1 = Transpose
// alpha: scaling factor applied during packing (use 1.0 for no scaling)
// Returns NULL on failure.
FbgemmPackedMatrix* fbgemm_packed_matrix_new(
    int trans,
    int nrow,
    int ncol,
    float alpha,
    const float* data);

void fbgemm_packed_matrix_free(FbgemmPackedMatrix* mat);

int fbgemm_packed_matrix_nrow(const FbgemmPackedMatrix* mat);
int fbgemm_packed_matrix_ncol(const FbgemmPackedMatrix* mat);

// Compute C = beta * C + A * packed_B
// A is m x k row-major, C is m x n row-major
// packed_B was created from a k x n matrix
void fbgemm_sgemm(
    int m,
    const float* A,
    const FbgemmPackedMatrix* packed_B,
    float beta,
    float* C,
    int num_threads);

#ifdef __cplusplus
}
#endif

#endif
