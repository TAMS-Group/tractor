// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/eigen.h>
#include <tractor/core/engine.h>

namespace tractor {

void buildGradientMatrix(const std::shared_ptr<Executable> &executable,
                         const std::shared_ptr<Memory> &memory,
                         Eigen::MatrixXd &matrix);

void buildGradientMatrix(const std::shared_ptr<Executable> &executable,
                         const std::shared_ptr<Memory> &memory,
                         Eigen::MatrixXf &matrix);

}  // namespace tractor
