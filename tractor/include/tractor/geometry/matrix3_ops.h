// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/operator.h>
#include <tractor/geometry/matrix3.h>
#include <tractor/geometry/vector3_ops.h>

namespace tractor {

TRACTOR_VAR_OP(inverse)

TRACTOR_OP_T(mat3, move, (const Matrix3<T> &v), { return Matrix3<T>(v); })
TRACTOR_D_T(prepare, mat3, move, (const Matrix3<T> &a, const Matrix3<T> &x), {})
TRACTOR_D_T(forward, mat3, move, (const Matrix3<T> &da, Matrix3<T> &dx),
            { dx = da; })
TRACTOR_D_T(reverse, mat3, move, (Matrix3<T> & da, const Matrix3<T> &dx),
            { da = dx; })

TRACTOR_OP_T(mat3_vec3, mul, (const Matrix3<T> &a, const Vector3<T> &b), {
  // return a * b;
  Vector3<T> x = Vector3<T>::Zero();
  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      x[row] += a(row, col) * b[col];
    }
  }
  return x;
})
TRACTOR_D_T(forward, mat3_vec3, mul,
            (const Matrix3<T> &pa, const Vector3<T> &pb, const Vector3<T> &px,
             const Matrix3<T> &da, const Vector3<T> &db, Vector3<T> &dx),
            {
              // dx = pa * db + da * pb;
              dx.setZero();
              for (size_t row = 0; row < 3; row++) {
                for (size_t col = 0; col < 3; col++) {
                  dx[row] += da(row, col) * pb[col];
                  dx[row] += pa(row, col) * db[col];
                }
              }
            })
TRACTOR_D_T(reverse, mat3_vec3, mul,
            (const Matrix3<T> &pa, const Vector3<T> &pb, const Vector3<T> &px,
             Matrix3<T> &da, Vector3<T> &db, const Vector3<T> &dx),
            {
              da.setZero();
              db.setZero();
              for (size_t row = 0; row < 3; row++) {
                for (size_t col = 0; col < 3; col++) {
                  da(row, col) += dx[row] * pb[col];
                  db[col] += dx[row] * pa(row, col);
                }
              }
            })

TRACTOR_OP_T(mat3, add, (const Matrix3<T> &a, const Matrix3<T> &b),
             { return a + b; })
TRACTOR_D_T(prepare, mat3, add,
            (const Matrix3<T> &a, const Matrix3<T> &b, const Matrix3<T> &x), {})
TRACTOR_D_T(forward, mat3, add,
            (const Matrix3<T> &da, const Matrix3<T> &db, Matrix3<T> &dx),
            { dx = da + db; })
TRACTOR_D_T(reverse, mat3, add,
            (Matrix3<T> & da, Matrix3<T> &db, const Matrix3<T> &dx), {
              da = dx;
              db = dx;
            })

TRACTOR_OP_T(mat3, sub, (const Matrix3<T> &a, const Matrix3<T> &b),
             { return a - b; })
TRACTOR_D_T(prepare, mat3, sub,
            (const Matrix3<T> &a, const Matrix3<T> &b, const Matrix3<T> &x), {})
TRACTOR_D_T(forward, mat3, sub,
            (const Matrix3<T> &da, const Matrix3<T> &db, Matrix3<T> &dx),
            { dx = da - db; })
TRACTOR_D_T(reverse, mat3, sub,
            (Matrix3<T> & da, Matrix3<T> &db, const Matrix3<T> &dx), {
              da = dx;
              db = -dx;
            })

TRACTOR_OP_T(mat3, zero, (Matrix3<T> & x), { x.setZero(); })
TRACTOR_D_T(prepare, mat3, zero, (const Matrix3<T> &x), {})
TRACTOR_D_T(forward, mat3, zero, (Matrix3<T> & dx), { dx.setZero(); })
TRACTOR_D_T(reverse, mat3, zero, (const Matrix3<T> &dx), {})

TRACTOR_OP_T(mat3, minus, (const Matrix3<T> &a), { return -a; })
TRACTOR_D_T(prepare, mat3, minus, (const Matrix3<T> &a, const Matrix3<T> &x),
            {})
TRACTOR_D_T(forward, mat3, minus, (const Matrix3<T> &da, Matrix3<T> &dx),
            { dx = -da; })
TRACTOR_D_T(reverse, mat3, minus, (Matrix3<T> & da, const Matrix3<T> &dx),
            { da = -dx; })

// template <class T, class M>
// static void invert_square_matrix(const M &mat, M &ret, size_t size) {
//   // for (size_t row = 0; row < size; row++) {
//   //   for (size_t col = 0; col < size; col++) {
//   //     if (row != col) {
//   //       ret(row, col) = T(0);
//   //     } else {
//   //       ret(row, col) = T(1);
//   //     }
//   //   }
//   // }
//   // for (size_t row = 0; row < size; row++) {
//   //   T f = T(1) / mat(row, row);
//   //   for (size_t col = 0; col < size; col++) {
//   //     mat(row, col) *= f;
//   //     ret(row, col) *= f;
//   //   }
//   // }
//   // for(size_t col = 0; col < size; col++) {
//   //
//   // }
// }

template <class T> Matrix3<T> transpose(const Matrix3<T> &mat) {
  Matrix3<T> ret;
  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      ret(col, row) = mat(row, col);
    }
  }
  return ret;
}

template <class T>
Matrix3<T> operator*(const Matrix3<T> &a, const Matrix3<T> &b) {
  Matrix3<T> x;
  auto ma = Eigen::Map<const Eigen::Matrix<T, 3, 3>>(a.data());
  auto mb = Eigen::Map<const Eigen::Matrix<T, 3, 3>>(b.data());
  auto mx = Eigen::Map<Eigen::Matrix<T, 3, 3>>(x.data());
  mx = ma * mb;
  return x;
}

template <class T> static Matrix3<T> inverse(const Matrix3<T> &mat) {
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

} // namespace tractor
