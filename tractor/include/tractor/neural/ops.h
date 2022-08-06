// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/factory.h>
#include <tractor/core/lambda.h>
#include <tractor/core/operator.h>
#include <tractor/core/profiler.h>

#include <random>

namespace tractor {

template <class Activation, class Bias>
Tensor<Activation> neural_bias(const Tensor<Activation> &inputs,
                               const Tensor<Bias> &bias) {

  if (bias.shape().dimensions() == 0 ||
      inputs.shape().dimensions() < bias.shape().dimensions()) {
    throw std::runtime_error("neural_bias invalid input shape");
  }

  for (size_t i = 0; i < bias.shape().dimensions(); i++) {
    if (inputs.shape().last(i) != bias.shape().last(i)) {
      throw std::runtime_error("neural_bias tensor shape mismatch");
    }
  }

  Tensor<Activation> outputs(inputs.shape());

  static Factory::Key<const TensorInfo *,
                      const TensorInfo *>::Value<const Operator *>
      factory{[](const TensorInfo *input_info, const TensorInfo *bias_info) {
        std::vector<Operator::Argument> args = {
            Operator::Argument::makeInput(input_info->type()),
            Operator::Argument::makeInput(bias_info->type()),
            Operator::Argument::makeOutput(input_info->type()),
        };
        for (auto &a : args) {
          TRACTOR_DEBUG_STREAM("tensor mul arg " << a.typeInfo().name());
        }

        std::string base_name = "neural_bias";
        std::string variant_name = std::string() + input_info->type().name() +
                                   "_" + bias_info->type().name();

        size_t inner = bias_info->shape().elementCount();
        size_t outer = input_info->shape().elementCount() /
                       bias_info->shape().elementCount();

        std::shared_ptr<ProfilerTrack> profiler_nonlinear =
            Profiler::instance()->track(std::make_shared<ProfilerTrack>(
                __PRETTY_FUNCTION__, base_name + "_" + variant_name + "_n"));

        std::shared_ptr<ProfilerTrack> profiler_forward =
            Profiler::instance()->track(std::make_shared<ProfilerTrack>(
                __PRETTY_FUNCTION__, base_name + "_" + variant_name + "_f"));

        std::shared_ptr<ProfilerTrack> profiler_reverse =
            Profiler::instance()->track(std::make_shared<ProfilerTrack>(
                __PRETTY_FUNCTION__, base_name + "_" + variant_name + "_r"));

        const Operator *op = makePointerOp(
            base_name, variant_name, args,

            [inner, outer, profiler_nonlinear](const Activation *a,
                                               const Bias *b, Activation *x)
                TRACTOR_FAST {
                  ProfilerScope profiler_scope(*profiler_nonlinear);
                  for (size_t o = 0; o < outer; o++) {
                    for (size_t i = 0; i < inner; i++) {
                      x[o * inner + i] = a[o * inner + i] + Activation(b[i]);
                    }
                  }
                },

            [inner, outer, profiler_forward](
                const Activation *a, const Bias *b, Activation *x,
                const Activation *da, const Bias *db, Activation *dx)
                TRACTOR_FAST {
                  ProfilerScope profiler_scope(*profiler_forward);
                  for (size_t o = 0; o < outer; o++) {
                    for (size_t i = 0; i < inner; i++) {
                      dx[o * inner + i] = da[o * inner + i] + Activation(db[i]);
                    }
                  }
                },

            [inner, outer, profiler_reverse](
                const Activation *a, const Bias *b, Activation *x,
                Activation *da, Bias *db, const Activation *dx) TRACTOR_FAST {
              ProfilerScope profiler_scope(*profiler_reverse);
              for (size_t i = 0; i < inner; i++) {
                Bias s = Bias(0);
                for (size_t o = 0; o < outer; o++) {
                  Activation a = dx[o * inner + i];
                  da[o * inner + i] = a;
                  Bias t = Bias(0);
                  batch_sum(a, t);
                  s += t;
                }
                db[i] = s;
              }
            });

        return op;
      }};

  auto *op = factory.get(inputs.info(), bias.info());

  std::array<void *, 3> args = {
      (void *)inputs.data(),
      (void *)bias.data(),
      (void *)outputs.data(),
  };
  callAndRecord(op, args.data());

  return outputs;
}

template <class Activation, class Weight>
Tensor<Activation> neural_dense(const Tensor<Activation> &inputs,
                                const Tensor<Weight> &weights) {

  if (weights.shape().dimensions() != 2) {
    throw std::runtime_error(
        "neural_dense invalid number of dimensions for weight matrix");
  }
  size_t input_neurons = weights.shape()[0];
  size_t output_neurons = weights.shape()[1];

  size_t batch_size = 0;
  size_t input_vector_neurons = 0;
  TensorShape output_shape;
  switch (inputs.shape().dimensions()) {
  case 1:
    batch_size = 1;
    input_vector_neurons = inputs.shape()[0];
    output_shape = TensorShape({output_neurons});
    break;
  case 2:
    batch_size = inputs.shape()[0];
    input_vector_neurons = inputs.shape()[1];
    output_shape = TensorShape({batch_size, output_neurons});
    break;
  default:
    throw std::runtime_error("neural_dense invalid input shape for "
                             "activation vector or batch of vectors");
  }
  if (input_vector_neurons != input_neurons) {
    throw std::runtime_error("neural_dense input vector length does not "
                             "match weight matrix size");
  }

  Tensor<Activation> outputs(output_shape);

  static Factory::Key<const TensorInfo *, const TensorInfo *,
                      const TensorInfo *, size_t, size_t,
                      size_t>::Value<const Operator *>
      factory{[](const TensorInfo *input_info, const TensorInfo *weight_info,
                 const TensorInfo *output_info, size_t input_neurons,
                 size_t output_neurons, size_t batch_size) {
        std::vector<Operator::Argument> args = {
            Operator::Argument::makeInput(input_info->type()),
            Operator::Argument::makeInput(weight_info->type()),
            Operator::Argument::makeOutput(output_info->type()),
        };
        for (auto &a : args) {
          TRACTOR_DEBUG_STREAM("tensor mul arg " << a.typeInfo().name());
        }

        std::string base_name = "neural_dense";
        std::string variant_name = std::string() + input_info->type().name() +
                                   "_" + weight_info->type().name();

        std::shared_ptr<ProfilerTrack> profiler_nonlinear =
            Profiler::instance()->track(std::make_shared<ProfilerTrack>(
                __PRETTY_FUNCTION__, base_name + "_" + variant_name + "_n"));

        std::shared_ptr<ProfilerTrack> profiler_forward =
            Profiler::instance()->track(std::make_shared<ProfilerTrack>(
                __PRETTY_FUNCTION__, base_name + "_" + variant_name + "_f"));

        std::shared_ptr<ProfilerTrack> profiler_reverse =
            Profiler::instance()->track(std::make_shared<ProfilerTrack>(
                __PRETTY_FUNCTION__, base_name + "_" + variant_name + "_r"));

        const Operator *op = makePointerOp(
            base_name, variant_name, args,

            [batch_size, input_neurons, output_neurons,
             profiler_nonlinear](const Activation *a, const Weight *b,
                                 Activation *x) TRACTOR_FAST {
              ProfilerScope profiler_scope(*profiler_nonlinear);
              for (size_t batch_index = 0; batch_index < batch_size;
                   batch_index++) {
                for (size_t output_neuron = 0; output_neuron < output_neurons;
                     output_neuron++) {
                  Activation v = Activation(0);
                  for (size_t input_neuron = 0; input_neuron < input_neurons;
                       input_neuron++) {
                    v += a[batch_index * input_neurons + input_neuron] *
                         Activation(
                             b[input_neuron * output_neurons + output_neuron]);
                  }
                  x[batch_index * output_neurons + output_neuron] = v;
                }
              }
            },

            [batch_size, input_neurons, output_neurons, profiler_forward](
                const Activation *a, const Weight *b, Activation *x,
                const Activation *da, const Weight *db, Activation *dx)
                TRACTOR_FAST {
                  ProfilerScope profiler_scope(*profiler_forward);
                  for (size_t batch_index = 0; batch_index < batch_size;
                       batch_index++) {
                    for (size_t output_neuron = 0;
                         output_neuron < output_neurons; output_neuron++) {
                      Activation dv = Activation(0);
                      for (size_t input_neuron = 0;
                           input_neuron < input_neurons; input_neuron++) {
                        dv += da[batch_index * input_neurons + input_neuron] *
                                  Activation(b[input_neuron * output_neurons +
                                               output_neuron]) +
                              a[batch_index * input_neurons + input_neuron] *
                                  Activation(db[input_neuron * output_neurons +
                                                output_neuron]);
                      }
                      dx[batch_index * output_neurons + output_neuron] = dv;
                    }
                  }
                },

            [batch_size, input_neurons, output_neurons, profiler_reverse](
                const Activation *a, const Weight *b, Activation *x,
                Activation *da, Weight *db, const Activation *dx) TRACTOR_FAST {
              ProfilerScope profiler_scope(*profiler_reverse);

              if (false) {
                {
                  auto *da_p = da;
                  for (size_t batch_index = 0; batch_index < batch_size;
                       batch_index++) {
                    for (size_t input_neuron = 0; input_neuron < input_neurons;
                         input_neuron++) {
                      Activation dv = Activation(0);
                      auto *dx_base = dx + batch_index * output_neurons;
                      auto *b_base = b + input_neuron * output_neurons;
                      for (size_t output_neuron = 0;
                           output_neuron < output_neurons; output_neuron++) {
                        dv += dx_base[output_neuron] *
                              Activation(b_base[output_neuron]);
                      }
                      *da_p = dv;
                      da_p++;
                    }
                  }
                }
                {
                  auto *db_p = db;
                  for (size_t input_neuron = 0; input_neuron < input_neurons;
                       input_neuron++) {
                    for (size_t output_neuron = 0;
                         output_neuron < output_neurons; output_neuron++) {
                      Weight w = Weight(0);
                      for (size_t batch_index = 0; batch_index < batch_size;
                           batch_index++) {
                        Weight v = Weight(0);
                        batch_sum(
                            dx[batch_index * output_neurons + output_neuron] *
                                a[batch_index * input_neurons + input_neuron],
                            v);
                        w += v;
                      }
                      *db_p = w;
                      db_p++;
                    }
                  }
                }
                return;
              }

              if (true) {
                for (size_t batch_index = 0; batch_index < batch_size;
                     batch_index++) {
                  for (size_t input_neuron = 0; input_neuron < input_neurons;
                       input_neuron++) {
                    Activation dv = Activation(0);
                    for (size_t output_neuron = 0;
                         output_neuron < output_neurons; output_neuron++) {
                      dv +=
                          dx[batch_index * output_neurons + output_neuron] *
                          Activation(
                              b[input_neuron * output_neurons + output_neuron]);
                    }
                    da[batch_index * input_neurons + input_neuron] = dv;
                  }
                }
                for (size_t input_neuron = 0; input_neuron < input_neurons;
                     input_neuron++) {
                  for (size_t output_neuron = 0; output_neuron < output_neurons;
                       output_neuron++) {
                    Weight w = Weight(0);
                    for (size_t batch_index = 0; batch_index < batch_size;
                         batch_index++) {
                      Weight v = Weight(0);
                      batch_sum(
                          dx[batch_index * output_neurons + output_neuron] *
                              a[batch_index * input_neurons + input_neuron],
                          v);
                      w += v;
                    }
                    db[input_neuron * output_neurons + output_neuron] = w;
                  }
                }
                return;
              }
            });

        return op;
      }};

  auto *op = factory.get(inputs.info(), weights.info(), outputs.info(),
                         input_neurons, output_neurons, batch_size);

  std::array<void *, 3> args = {
      (void *)inputs.data(),
      (void *)weights.data(),
      (void *)outputs.data(),
  };
  callAndRecord(op, args.data());

  return outputs;
}

// ------------------------------------------

template <class T> T add_random_normal(const T &a, const T &s) {
  static thread_local std::mt19937 rng{std::mt19937::result_type(rand())};
  std::normal_distribution<double> dist;
  return a + T(dist(rng)) * s;
}
template <class T, size_t S>
Batch<T, S> add_random_normal(const Batch<T, S> &a, const T &s) {
  Batch<T, S> ret;
  for (size_t i = 0; i < S; i++) {
    ret[i] = add_random_normal(a[i], s);
  }
  return ret;
}
TRACTOR_OP(add_random_normal, (const T &a, const S &s),
           { return add_random_normal(a, s); })
TRACTOR_D(prepare, add_random_normal, (const T &a, const S &s, const T &x), {})
TRACTOR_D(forward, add_random_normal, (const T &da, const S &ds, T &dx),
          { dx = da; })
TRACTOR_D(reverse, add_random_normal, (T & da, S &ds, const T &dx), {
  da = dx;
  ds = S(0);
})

// ------------------------------------------

template <class T> T add_random_uniform(const T &a, const T &l, const T &h) {
  static thread_local std::mt19937 rng{std::mt19937::result_type(rand())};
  std::uniform_real_distribution<double> dist(l, h);
  return a + T(dist(rng));
}
template <class T, size_t S>
Batch<T, S> add_random_uniform(const Batch<T, S> &a, const T &l, const T &h) {
  Batch<T, S> ret;
  for (size_t i = 0; i < S; i++) {
    ret[i] = add_random_uniform(a[i], l, h);
  }
  return ret;
}
TRACTOR_OP(add_random_uniform, (const T &a, const S &l, const S &h),
           { return add_random_uniform(a, l, h); })
TRACTOR_D(prepare, add_random_uniform,
          (const T &a, const S &l, const S &h, const T &x), {})
TRACTOR_D(forward, add_random_uniform,
          (const T &da, const S &dl, const S &dh, T &dx), { dx = da; })
TRACTOR_D(reverse, add_random_uniform, (T & da, S &dl, S &dh, const T &dx), {
  da = dx;
  dl = S(0);
  dh = S(0);
})

// ------------------------------------------

template <class T> inline void dropout3(const T &a, const T &b, T &x, T &y) {
  static thread_local std::mt19937 rng{std::mt19937::result_type(rand())};
  std::uniform_real_distribution<double> dist;
  y = (dist(rng) < b) ? T(0) : T(1.0 / (1.0 - b));
  x = a * y;
}
template <class T, size_t S>
inline void dropout3(const Batch<T, S> &a, const T &b, Batch<T, S> &x,
                     Batch<T, S> &y) {
  for (size_t i = 0; i < S; i++) {
    dropout3(a[i], b, x[i], y[i]);
  }
}
TRACTOR_OP(dropout2, (const T &a, const S &b, T &x, T &y),
           { dropout3(a, b, x, y); })
TRACTOR_D(prepare, dropout2,
          (const T &a, const S &b, const T &x, const T &y, T &s), { s = y; })
TRACTOR_D(forward, dropout2,
          (const T &s, const T &da, const S &db, T &dx, T &dy), {
            dx = da * s;
            dy = T(0);
          })
TRACTOR_D(reverse, dropout2,
          (const T &s, T &da, S &db, const T &dx, const T &dy), {
            da = dx * s;
            db = S(0);
          })

template <class A, class B> A dropout(const A &a, const B &b) {
  A x(a.shape());
  A y(a.shape());
  dropout2(a, b, x, y);
  return x;
}

// ------------------------------------------

TRACTOR_OP(relu, (const T &a), { return std::max(T(0), a); })
TRACTOR_D_LOOP(forward, relu, (const T &a, const T &x, const T &da, T &dx),
               (a, x, da, dx), {
                 if (a >= T(0)) {
                   dx = da;
                 } else {
                   dx = T(0);
                 }
               })
TRACTOR_D_LOOP(reverse, relu, (const T &a, const T &x, T &da, const T &dx),
               (a, x, da, dx), {
                 if (a >= T(0)) {
                   da = dx;
                 } else {
                   da = T(0);
                 }
               })

} // namespace tractor
