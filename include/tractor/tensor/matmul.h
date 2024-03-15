// 2020-2024 Philipp Ruppel

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

}  // namespace Eigen

namespace tractor {

template <class A, class X>
inline void _internal_matmul_add(const A &a, X &x) {
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

  for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
    for (size_t output_neuron = 0; output_neuron < output_neurons;
         output_neuron++) {
      X v = X(0);
      for (size_t input_neuron = 0; input_neuron < input_neurons;
           input_neuron++) {
        v += X(a[batch_index * input_neurons + input_neuron] *
               b[input_neuron * output_neurons + output_neuron]);
      }
      x[batch_index * output_neurons + output_neuron] = v;
    }
  }
}

template <class A, class B, class X>
void matmul_forward(size_t batch_size, size_t input_neurons,
                    size_t output_neurons, const A *a, const B *b, const X *x,
                    const A *da, const B *db, X *dx) {
  TRACTOR_PROFILER("matmul forward");

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

}  // namespace tractor
