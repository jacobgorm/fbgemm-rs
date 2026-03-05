#include "fbgemm_sgemm.h"
#include "fbgemm/FbgemmFP32.h"

using fbgemm::PackedGemmMatrixB;
using fbgemm::matrix_op_t;

extern "C" {

FbgemmPackedMatrix* fbgemm_packed_matrix_new(
    int trans,
    int nrow,
    int ncol,
    float alpha,
    const float* data) {
  try {
    auto op = (trans == 0) ? matrix_op_t::NoTranspose : matrix_op_t::Transpose;
    auto* mat = new PackedGemmMatrixB<float>(op, nrow, ncol, alpha, data);
    return reinterpret_cast<FbgemmPackedMatrix*>(mat);
  } catch (...) {
    return nullptr;
  }
}

void fbgemm_packed_matrix_free(FbgemmPackedMatrix* mat) {
  delete reinterpret_cast<PackedGemmMatrixB<float>*>(mat);
}

int fbgemm_packed_matrix_nrow(const FbgemmPackedMatrix* mat) {
  return reinterpret_cast<const PackedGemmMatrixB<float>*>(mat)->numRows();
}

int fbgemm_packed_matrix_ncol(const FbgemmPackedMatrix* mat) {
  return reinterpret_cast<const PackedGemmMatrixB<float>*>(mat)->numCols();
}

void fbgemm_sgemm(
    int m,
    const float* A,
    const FbgemmPackedMatrix* packed_B,
    float beta,
    float* C,
    int num_threads) {
  const auto& Bp =
      *reinterpret_cast<const PackedGemmMatrixB<float>*>(packed_B);

  if (num_threads <= 1) {
    fbgemm::cblas_gemm_compute(
        matrix_op_t::NoTranspose, m, A, Bp, beta, C, 0, 1);
  } else {
    #pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < num_threads; ++t) {
      fbgemm::cblas_gemm_compute(
          matrix_op_t::NoTranspose, m, A, Bp, beta, C, t, num_threads);
    }
  }
}

} // extern "C"
