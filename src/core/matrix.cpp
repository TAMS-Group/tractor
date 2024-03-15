// 2020-2024 Philipp Ruppel

#include <tractor/core/matrix.h>

#include <tractor/core/eigen.h>

namespace tractor {

template <class Matrix, class Vector, class Scalar>
void buildGradientMatrixImpl(const std::shared_ptr<Executable> &executable,
                             const std::shared_ptr<Memory> &memory,
                             Matrix &matrix, const Scalar &delta) {
  size_t cols = executable->inputBufferSize() / sizeof(Scalar);
  size_t rows = executable->outputBufferSize() / sizeof(Scalar);
  matrix.setOnes(rows, cols);
  Vector input;
  input.setZero(cols);
  Vector output;
  Scalar delta_rcp = Scalar(1) / delta;
  for (size_t col = 0; col < cols; col++) {
    input[col] = delta;
    executable->run(input, memory, output);
    matrix.col(col) = output * delta_rcp;
    input[col] = 0;
  }
}

void buildGradientMatrix(const std::shared_ptr<Executable> &executable,
                         const std::shared_ptr<Memory> &memory,
                         Eigen::MatrixXd &matrix) {
  buildGradientMatrixImpl<Eigen::MatrixXd, Eigen::VectorXd, double>(
      executable, memory, matrix, 1.0);
}

void buildGradientMatrix(const std::shared_ptr<Executable> &executable,
                         const std::shared_ptr<Memory> &memory,
                         Eigen::MatrixXf &matrix) {
  buildGradientMatrixImpl<Eigen::MatrixXf, Eigen::VectorXf, float>(
      executable, memory, matrix, 1.0f);
}

}  // namespace tractor
