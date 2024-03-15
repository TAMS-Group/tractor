// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/eigen.h>
#include <tractor/core/operator.h>
#include <tractor/geometry/eigen.h>
#include <tractor/geometry/matrix3.h>
#include <tractor/geometry/matrix3_ops.h>

namespace tractor {

template <class T>
Matrix3<T> operator*(const Matrix3<T> &a, const Matrix3<T> &b) {
  Matrix3<T> x;
  auto ma = Eigen::Map<const Eigen::Matrix<T, 3, 3>>(a.data());
  auto mb = Eigen::Map<const Eigen::Matrix<T, 3, 3>>(b.data());
  auto mx = Eigen::Map<Eigen::Matrix<T, 3, 3>>(x.data());
  mx = ma * mb;
  return x;
}

template <class T>
static Matrix3<T> inverse(const Matrix3<T> &mat) {
  Matrix3<T> ret;
  auto mret = Eigen::Map<Eigen::Matrix<T, 3, 3>>(ret.data());
  auto mmat = Eigen::Map<const Eigen::Matrix<T, 3, 3>>(mat.data());
  mret = mmat.inverse();
  // mret = mmat.llt().solve(Eigen::Matrix<T, 3, 3>::Identity());
  return ret;
}

TRACTOR_OP_T(mat3, inverse, (const Matrix3<T> &a), { return inverse(a); })
TRACTOR_D_T(prepare, mat3, inverse,
            (const Matrix3<T> &a, const Matrix3<T> &x, Matrix3<T> &v),
            { v = x; })
TRACTOR_D_T(forward, mat3, inverse,
            (const Matrix3<T> &v, const Matrix3<T> &da, Matrix3<T> &dx),
            { dx = -(v * da * v); })
TRACTOR_D_T(reverse, mat3, inverse,
            (const Matrix3<T> &v, Matrix3<T> &da, const Matrix3<T> &dx),
            { da = (transpose(v) * (-dx) * transpose(v)); })

}  // namespace tractor
