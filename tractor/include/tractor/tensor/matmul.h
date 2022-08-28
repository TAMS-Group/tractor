// (c) 2020-2022 Philipp Ruppel

#pragma once

// #define EIGEN_GEMM_TO_COEFFBASED_THRESHOLD 1

#include <tractor/core/eigen.h>
#include <tractor/core/profiler.h>

namespace Eigen {

template <class Scalar, size_t BatchSize, class BinaryOp>
struct ScalarBinaryOpTraits<tractor::Batch<Scalar, BatchSize>, Scalar,
                            BinaryOp> {
  typedef tractor::Batch<Scalar, BatchSize> ReturnType;
};

template <class Scalar, size_t BatchSize, class BinaryOp>
struct ScalarBinaryOpTraits<Scalar, tractor::Batch<Scalar, BatchSize>,
                            BinaryOp> {
  typedef tractor::Batch<Scalar, BatchSize> ReturnType;
};

template <class Scalar, size_t BatchSize, class BinaryOp>
struct ScalarBinaryOpTraits<tractor::Batch<Scalar, BatchSize>,
                            tractor::Batch<Scalar, BatchSize>, BinaryOp> {
  typedef tractor::Batch<Scalar, BatchSize> ReturnType;
};

} // namespace Eigen

namespace tractor {

//#define CHECK_TENSOR_MATMUL

// template <class A, class X> struct MatMulCast {
//   const X cast(const A &a) { return X(a); }
// };
// template <class A, class X, class S> struct MatMulCast<A, Batch<X, S>> {
//   const Batch<X, S> cast(const A &a) { return Batch<X, S>(a); }
// };
// template <class A, class X, class S> struct MatMulCast<Batch<A, S>, X> {
//   const X cast(const Batch<A, S> &a) { return batch_sum(a); }
// };
// template <class A, class X, class S>
// struct MatMulCast<Batch<A, S>, Batch<X, S>> {
//   const Batch<X, S> cast(const Batch<A, S> &a) { return Batch<X, S>(a); }
// };

template <class A, class X> inline void _internal_matmul_add(const A &a, X &x) {
  x += a;
}

template <class A, class X, size_t S>
inline void _internal_matmul_add(const A &a, Batch<X, S> &x) {
  x += X(a);
}

template <class A, class X, size_t S>
inline void _internal_matmul_add(const Batch<A, S> &a, X &x) {
  x += batch_sum(a);
}

template <class A, class X, size_t S>
inline void _internal_matmul_add(const Batch<A, S> &a, Batch<X, S> &x) {
  x += Batch<X, S>(a);
}

template <class A, class B, class X>
void matmul_compute(size_t batch_size, size_t input_neurons,
                    size_t output_neurons, const A *a, const B *b, X *x) {

  TRACTOR_PROFILER("matmul compute");

  // for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
  //   for (size_t output_neuron = 0; output_neuron < output_neurons;
  //        output_neuron++) {
  //     X v = X(0);
  //     for (size_t input_neuron = 0; input_neuron < input_neurons;
  //          input_neuron++) {
  //       v += X(a[batch_index * input_neurons + input_neuron] *
  //              b[input_neuron * output_neurons + output_neuron]);
  //     }
  //     x[batch_index * output_neurons + output_neuron] = v;
  //   }
  // }

  if (true) {
    Eigen::internal::set_is_malloc_allowed(false);
    auto ma = Eigen::Map<
        const Eigen::Matrix<A, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        // , Eigen::Unaligned
        >(a, batch_size, input_neurons);
    auto mb = Eigen::Map<
        const Eigen::Matrix<B, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        // , Eigen::Unaligned
        >(b, input_neurons, output_neurons);
    auto mx = Eigen::Map<
        Eigen::Matrix<X, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        // , Eigen::Unaligned
        >(x, batch_size, output_neurons);
    // mx.noalias() = ma.cast<X>() * mb.cast<X>();
    // mx.noalias() = ma * mb;
    // mx.noalias() = ma.lazyProduct(mb);
    // mx.noalias() = ma.unaryExpr([](const A &a) { return X(a); }) *
    //                mb.unaryExpr([](const B &b) { return X(b); });
    auto xma = ma.unaryExpr([](const A &a) { return X(a); });
    auto xmb = mb.unaryExpr([](const B &b) { return X(b); });
    mx.noalias() = xma.lazyProduct(xmb);
    Eigen::internal::set_is_malloc_allowed(true);
  }

  // #ifdef CHECK_TENSOR_MATMUL
  //   for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
  //     for (size_t output_neuron = 0; output_neuron < output_neurons;
  //          output_neuron++) {
  //       X v = X(0);
  //       for (size_t input_neuron = 0; input_neuron < input_neurons;
  //            input_neuron++) {
  //         v += a[batch_index * input_neurons + input_neuron] *
  //              X(b[input_neuron * output_neurons + output_neuron]);
  //       }
  //       TRACTOR_ASSERT(
  //           std::abs(x[batch_index * output_neurons + output_neuron] - v) <
  //           1e-9);
  //     }
  //   }
  // #endif
}

template <class A, class B, class X>
void matmul_forward(size_t batch_size, size_t input_neurons,
                    size_t output_neurons, const A *a, const B *b, const X *x,
                    const A *da, const B *db, X *dx) {

  TRACTOR_PROFILER("matmul forward");

  //   if (true) {
  //     // TRACTOR_INFO("pointer a " << a);
  //     // TRACTOR_INFO("pointer b " << b);
  //     // TRACTOR_INFO("pointer da " << da);
  //     // TRACTOR_INFO("pointer db " << db);
  //     // TRACTOR_INFO("pointer dx " << dx);
  //     {
  //       auto ma = Eigen::Map<const Eigen::Matrix<A, Eigen::Dynamic,
  //                                                Eigen::Dynamic,
  //                                                Eigen::RowMajor>,
  //                            Eigen::Unaligned>(a, batch_size, input_neurons);
  //       auto mb = Eigen::Map<const Eigen::Matrix<B, Eigen::Dynamic,
  //                                                Eigen::Dynamic,
  //                                                Eigen::RowMajor>,
  //                            Eigen::Unaligned>(b, input_neurons,
  //                            output_neurons);
  //       auto mda =
  //           Eigen::Map<const Eigen::Matrix<A, Eigen::Dynamic, Eigen::Dynamic,
  //                                          Eigen::RowMajor>,
  //                      Eigen::Unaligned>(da, batch_size, input_neurons);
  //       auto mdb =
  //           Eigen::Map<const Eigen::Matrix<B, Eigen::Dynamic, Eigen::Dynamic,
  //                                          Eigen::RowMajor>,
  //                      Eigen::Unaligned>(db, input_neurons, output_neurons);
  //       auto mdx = Eigen::Map<
  //           Eigen::Matrix<X, Eigen::Dynamic, Eigen::Dynamic,
  //           Eigen::RowMajor>, Eigen::Unaligned>(dx, batch_size,
  //           output_neurons);
  //       // mdx.noalias() = (ma * mdb + mda * mb);
  //       mdx.noalias() = ma * mdb;
  //       mdx.noalias() += mda * mb;
  //     }
  // #ifdef CHECK_TENSOR_MATMUL
  //     bool error = false;
  //     for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
  //       for (size_t output_neuron = 0; output_neuron < output_neurons;
  //            output_neuron++) {
  //         X dv = X(0);
  //         for (size_t input_neuron = 0; input_neuron < input_neurons;
  //              input_neuron++) {
  //           dv += da[batch_index * input_neurons + input_neuron] *
  //                     X(b[input_neuron * output_neurons + output_neuron]) +
  //                 a[batch_index * input_neurons + input_neuron] *
  //                     X(db[input_neuron * output_neurons + output_neuron]);
  //         }
  //         if (!(abs(dx[batch_index * output_neurons + output_neuron] - dv) <=
  //               1e-9)) {
  //           error = true;
  //         }
  //       }
  //     }
  //     if (error) {
  //       for (size_t batch_index = 0; batch_index < batch_size; batch_index++)
  //       {
  //         for (size_t output_neuron = 0; output_neuron < output_neurons;
  //              output_neuron++) {
  //           X dv = X(0);
  //           for (size_t input_neuron = 0; input_neuron < input_neurons;
  //                input_neuron++) {
  //             dv += da[batch_index * input_neurons + input_neuron] *
  //                       X(b[input_neuron * output_neurons + output_neuron]) +
  //                   a[batch_index * input_neurons + input_neuron] *
  //                       X(db[input_neuron * output_neurons + output_neuron]);
  //           }
  //           std::cout << dx[batch_index * output_neurons + output_neuron] <<
  //           "/"
  //                     << dv << " ";
  //         }
  //         std::cout << "\n";
  //       }
  //       std::cout << "\n";
  //       throw std::runtime_error("matmul error");
  //     }
  // #endif
  //   }

  for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
    for (size_t output_neuron = 0; output_neuron < output_neurons;
         output_neuron++) {
      X dv = X(0);
      for (size_t input_neuron = 0; input_neuron < input_neurons;
           input_neuron++) {
        dv += X(da[batch_index * input_neurons + input_neuron] *
                    b[input_neuron * output_neurons + output_neuron] +
                a[batch_index * input_neurons + input_neuron] *
                    db[input_neuron * output_neurons + output_neuron]);
      }
      dx[batch_index * output_neurons + output_neuron] = dv;
    }
  }
}

template <class A, class B, class X>
void matmul_reverse(size_t batch_size, size_t input_neurons,
                    size_t output_neurons, const A *a, const B *b, const X *x,
                    A *da, B *db, const X *dx) {

  // if (false) {
  //   {
  //     auto *da_p = da;
  //     for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
  //       for (size_t input_neuron = 0; input_neuron < input_neurons;
  //            input_neuron++) {
  //         Activation dv = Activation(0);
  //         auto *dx_base = dx + batch_index * output_neurons;
  //         auto *b_base = b + input_neuron * output_neurons;
  //         for (size_t output_neuron = 0; output_neuron < output_neurons;
  //              output_neuron++) {
  //           dv += dx_base[output_neuron] * Activation(b_base[output_neuron]);
  //         }
  //         *da_p = dv;
  //         da_p++;
  //       }
  //     }
  //   }
  //   {
  //     auto *db_p = db;
  //     for (size_t input_neuron = 0; input_neuron < input_neurons;
  //          input_neuron++) {
  //       for (size_t output_neuron = 0; output_neuron < output_neurons;
  //            output_neuron++) {
  //         Weight w = Weight(0);
  //         for (size_t batch_index = 0; batch_index < batch_size;
  //              batch_index++) {
  //           Weight v = Weight(0);
  //           batch_sum(dx[batch_index * output_neurons + output_neuron] *
  //                         a[batch_index * input_neurons + input_neuron],
  //                     v);
  //           w += v;
  //         }
  //         *db_p = w;
  //         db_p++;
  //       }
  //     }
  //   }
  // }
  //
  // if (false) {
  //   {
  //     TRACTOR_PROFILER("matmul backprop left");
  //     for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
  //       for (size_t input_neuron = 0; input_neuron < input_neurons;
  //            input_neuron++) {
  //         Activation dv = Activation(0);
  //         for (size_t output_neuron = 0; output_neuron < output_neurons;
  //              output_neuron++) {
  //           dv += dx[batch_index * output_neurons + output_neuron] *
  //                 Activation(b[input_neuron * output_neurons +
  //                 output_neuron]);
  //         }
  //         da[batch_index * input_neurons + input_neuron] = dv;
  //       }
  //     }
  //   }
  //   {
  //     TRACTOR_PROFILER("matmul backprop right");
  //     for (size_t input_neuron = 0; input_neuron < input_neurons;
  //          input_neuron++) {
  //       for (size_t output_neuron = 0; output_neuron < output_neurons;
  //            output_neuron++) {
  //         Weight w = Weight(0);
  //         for (size_t batch_index = 0; batch_index < batch_size;
  //              batch_index++) {
  //           Weight v = Weight(0);
  //           batch_sum(dx[batch_index * output_neurons + output_neuron] *
  //                         a[batch_index * input_neurons + input_neuron],
  //                     v);
  //           w += v;
  //         }
  //         db[input_neuron * output_neurons + output_neuron] = w;
  //       }
  //     }
  //   }
  // }
  //
  // if (false) {
  //   {
  //     TRACTOR_PROFILER("matmul backprop left");
  //     for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
  //       for (size_t input_neuron = 0; input_neuron < input_neurons;
  //            input_neuron++) {
  //         Activation dv = Activation(0);
  //         for (size_t output_neuron = 0; output_neuron < output_neurons;
  //              output_neuron++) {
  //           dv += dx[batch_index * output_neurons + output_neuron] *
  //                 Activation(b[input_neuron * output_neurons +
  //                 output_neuron]);
  //         }
  //         da[batch_index * input_neurons + input_neuron] = dv;
  //       }
  //     }
  //   }
  //   {
  //     TRACTOR_PROFILER("matmul backprop right");
  //     for (size_t input_neuron = 0; input_neuron < input_neurons;
  //          input_neuron++) {
  //       for (size_t output_neuron = 0; output_neuron < output_neurons;
  //            output_neuron++) {
  //         Weight w = Weight(0);
  //         for (size_t batch_index = 0; batch_index < batch_size;
  //              batch_index++) {
  //           w += batch_sum(dx[batch_index * output_neurons + output_neuron] *
  //                          a[batch_index * input_neurons + input_neuron]);
  //         }
  //         db[input_neuron * output_neurons + output_neuron] = w;
  //       }
  //     }
  //   }
  // }
  //
  // if (false) {
  //   // {
  //   //   TRACTOR_PROFILER("matmul backprop left");
  //   //   for (size_t batch_index = 0; batch_index < batch_size;
  //   //        batch_index++) {
  //   //     for (size_t input_neuron = 0; input_neuron <
  //   //     input_neurons;
  //   //          input_neuron++) {
  //   //       Activation dv = Activation(0);
  //   //       for (size_t output_neuron = 0;
  //   //            output_neuron < output_neurons; output_neuron++) {
  //   //         dv += dx[batch_index * output_neurons +
  //   //         output_neuron] *
  //   //               Activation(b[input_neuron * output_neurons +
  //   //                            output_neuron]);
  //   //       }
  //   //       da[batch_index * input_neurons + input_neuron] = dv;
  //   //     }
  //   //   }
  //   // }
  //
  //   {
  //     TRACTOR_PROFILER("matmul backprop left");
  //     auto mb = Eigen::Map<const Eigen::Matrix<Weight, Eigen::Dynamic,
  //                                              Eigen::Dynamic,
  //                                              Eigen::RowMajor>,
  //                          Eigen::Unaligned>(b, input_neurons,
  //                          output_neurons);
  //     auto mda = Eigen::Map<Eigen::Matrix<Activation, Eigen::Dynamic,
  //                                         Eigen::Dynamic, Eigen::RowMajor>,
  //                           Eigen::Unaligned>(da, batch_size, input_neurons);
  //     auto mdx =
  //         Eigen::Map<const Eigen::Matrix<Activation, Eigen::Dynamic,
  //                                        Eigen::Dynamic, Eigen::RowMajor>,
  //                    Eigen::Unaligned>(dx, batch_size, output_neurons);
  //     mda = mdx * mb.transpose();
  //   }
  //
  //   // {
  //   //   TRACTOR_PROFILER("check dense backprop left");
  //   //   for (size_t batch_index = 0; batch_index < batch_size;
  //   //        batch_index++) {
  //   //     for (size_t input_neuron = 0; input_neuron <
  //   //     input_neurons;
  //   //          input_neuron++) {
  //   //       Activation dv = Activation(0);
  //   //       for (size_t output_neuron = 0;
  //   //            output_neuron < output_neurons; output_neuron++) {
  //   //         dv += dx[batch_index * output_neurons +
  //   //         output_neuron] *
  //   //               Activation(b[input_neuron * output_neurons +
  //   //                            output_neuron]);
  //   //       }
  //   //       TRACTOR_ASSERT(
  //   //           std::abs(
  //   //               da[batch_index * input_neurons + input_neuron]
  //   //               - dv) < 1e-9);
  //   //     }
  //   //   }
  //   // }
  //
  //   {
  //     TRACTOR_PROFILER("matmul backprop zero");
  //     std::memset(db, 0, sizeof(Weight) * input_neurons * output_neurons);
  //   }
  //
  //   // {
  //   //   TRACTOR_PROFILER("matmul backprop right");
  //   //   for (size_t input_neuron = 0; input_neuron < input_neurons;
  //   //        input_neuron++) {
  //   //     for (size_t output_neuron = 0;
  //   //          output_neuron < output_neurons; output_neuron++) {
  //   //       for (size_t batch_index = 0; batch_index < batch_size;
  //   //            batch_index++) {
  //   //         db[input_neuron * output_neurons + output_neuron] +=
  //   //             batch_sum(
  //   //                 dx[batch_index * output_neurons +
  //   //                    output_neuron] *
  //   //                 a[batch_index * input_neurons +
  //   //                 input_neuron]);
  //   //       }
  //   //     }
  //   //   }
  //   // }
  //
  //   // {
  //   //   TRACTOR_PROFILER("matmul backprop right");
  //   //   for (size_t input_neuron = 0; input_neuron < input_neurons;
  //   //        input_neuron++) {
  //   //     for (size_t batch_index = 0; batch_index < batch_size;
  //   //          batch_index++) {
  //   //       for (size_t output_neuron = 0;
  //   //            output_neuron < output_neurons; output_neuron++) {
  //   //         db[input_neuron * output_neurons + output_neuron] +=
  //   //             batch_sum(
  //   //                 dx[batch_index * output_neurons +
  //   //                    output_neuron] *
  //   //                 a[batch_index * input_neurons +
  //   //                 input_neuron]);
  //   //       }
  //   //     }
  //   //   }
  //   // }
  //
  //   {
  //     TRACTOR_PROFILER("matmul backprop right");
  //     for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
  //       for (size_t input_neuron = 0; input_neuron < input_neurons;
  //            input_neuron++) {
  //         for (size_t output_neuron = 0; output_neuron < output_neurons;
  //              output_neuron++) {
  //           db[input_neuron * output_neurons + output_neuron] +=
  //               batch_sum(dx[batch_index * output_neurons + output_neuron] *
  //                         a[batch_index * input_neurons + input_neuron]);
  //         }
  //       }
  //     }
  //   }
  // }

  //   if (true) {
  //     {
  //       // TRACTOR_PROFILER("matmul backprop left");
  //       auto mb = Eigen::Map<const Eigen::Matrix<B, Eigen::Dynamic,
  //                                                Eigen::Dynamic,
  //                                                Eigen::RowMajor>,
  //                            Eigen::Unaligned>(b, input_neurons,
  //                            output_neurons);
  //       auto mda = Eigen::Map<
  //           Eigen::Matrix<A, Eigen::Dynamic, Eigen::Dynamic,
  //           Eigen::RowMajor>, Eigen::Unaligned>(da, batch_size,
  //           input_neurons);
  //       auto mdx =
  //           Eigen::Map<const Eigen::Matrix<X, Eigen::Dynamic, Eigen::Dynamic,
  //                                          Eigen::RowMajor>,
  //                      Eigen::Unaligned>(dx, batch_size, output_neurons);
  //       mda.noalias() = mdx * mb.transpose();
  //     }
  // #ifdef CHECK_TENSOR_MATMUL
  //     TRACTOR_PROFILER("check dense backprop left");
  //     for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
  //       for (size_t input_neuron = 0; input_neuron < input_neurons;
  //            input_neuron++) {
  //         A dv = A(0);
  //         for (size_t output_neuron = 0; output_neuron < output_neurons;
  //              output_neuron++) {
  //           dv += dx[batch_index * output_neurons + output_neuron] *
  //                 A(b[input_neuron * output_neurons + output_neuron]);
  //         }
  //         TRACTOR_ASSERT(std::abs(da[batch_index * input_neurons +
  //         input_neuron] -
  //                                 dv) < 1e-9);
  //       }
  //     }
  // #endif
  //     {
  //       // TRACTOR_PROFILER("matmul backprop right");
  //       auto ma = Eigen::Map<const Eigen::Matrix<A, Eigen::Dynamic,
  //                                                Eigen::Dynamic,
  //                                                Eigen::RowMajor>,
  //                            Eigen::Unaligned>(a, batch_size, input_neurons);
  //       auto mdb = Eigen::Map<
  //           Eigen::Matrix<B, Eigen::Dynamic, Eigen::Dynamic,
  //           Eigen::RowMajor>, Eigen::Unaligned>(db, input_neurons,
  //           output_neurons);
  //       auto mdx =
  //           Eigen::Map<const Eigen::Matrix<X, Eigen::Dynamic, Eigen::Dynamic,
  //                                          Eigen::RowMajor>,
  //                      Eigen::Unaligned>(dx, batch_size, output_neurons);
  //       // mdb.noalias() = ma.transpose() * mdx;
  //       // mdb.noalias() = (ma.transpose() * mdx).unaryExpr([](const
  //       Activation
  //       // &a) {
  //       //   return batch_sum(a);
  //       // });
  //     }
  // #ifdef CHECK_TENSOR_MATMUL
  //     {
  //       TRACTOR_PROFILER("check dense backprop right");
  //       for (size_t input_neuron = 0; input_neuron < input_neurons;
  //            input_neuron++) {
  //         for (size_t output_neuron = 0; output_neuron < output_neurons;
  //              output_neuron++) {
  //           B w = B(0);
  //           for (size_t batch_index = 0; batch_index < batch_size;
  //                batch_index++) {
  //             w += batch_sum(dx[batch_index * output_neurons + output_neuron]
  //             *
  //                            a[batch_index * input_neurons + input_neuron]);
  //           }
  //           TRACTOR_ASSERT(
  //               std::abs(db[input_neuron * output_neurons + output_neuron] -
  //               w) < 1e-9);
  //         }
  //       }
  //     }
  // #endif
  //   }

  {
    TRACTOR_PROFILER("matmul backprop left");
    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
      for (size_t input_neuron = 0; input_neuron < input_neurons;
           input_neuron++) {
        A dv = A(0);
        for (size_t output_neuron = 0; output_neuron < output_neurons;
             output_neuron++) {
          dv += A(dx[batch_index * output_neurons + output_neuron] *
                  b[input_neuron * output_neurons + output_neuron]);
        }
        da[batch_index * input_neurons + input_neuron] = dv;
      }
    }
  }
  {
    TRACTOR_PROFILER("matmul backprop right");
    size_t input_neuron_output_neurons = 0;
    for (size_t input_neuron = 0; input_neuron < input_neurons;
         input_neuron++) {
      for (size_t output_neuron = 0; output_neuron < output_neurons;
           output_neuron++) {
        B w = B(0);
        size_t batch_index_input_neurons = 0;
        size_t batch_index_output_neurons = 0;
        for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
          // Weight v = Weight(0);
          // batch_sum(dx[batch_index_output_neurons + output_neuron] *
          //               a[batch_index_input_neurons + input_neuron],
          //           v);
          // w += v;
          _internal_matmul_add(dx[batch_index_output_neurons + output_neuron] *
                                   a[batch_index_input_neurons + input_neuron],
                               w);
          batch_index_input_neurons += input_neurons;
          batch_index_output_neurons += output_neurons;
        }
        db[input_neuron_output_neurons + output_neuron] = w;
      }
      input_neuron_output_neurons += output_neurons;
    }
  }
}

} // namespace tractor
