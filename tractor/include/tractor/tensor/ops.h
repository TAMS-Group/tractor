// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/error.h>
#include <tractor/core/factory.h>
#include <tractor/core/list.h>
#include <tractor/core/ops.h>
#include <tractor/core/profiler.h>

#define TRACTOR_CHECK_TENSOR_DIMENSIONS(module, input, dims)                   \
  {                                                                            \
    if (input.shape().dimensions() != dims) {                                  \
      throw std::invalid_argument(                                             \
          std::string() + module +                                             \
          ": wrong number of dimensions, expected:" + std::to_string(dims) +   \
          ", found:" + std::to_string(input.shape().dimensions()));            \
    }                                                                          \
  }

namespace tractor {

template <class T> Tensor<T> operator+(const Tensor<T> &a, const Tensor<T> &b) {
  return add(a, b);
}
template <class T> Tensor<T> operator-(const Tensor<T> &a, const Tensor<T> &b) {
  return sub(a, b);
}
template <class T> Tensor<T> operator*(const Tensor<T> &a, const Tensor<T> &b) {
  return mul(a, b);
}
template <class T> Tensor<T> operator/(const Tensor<T> &a, const Tensor<T> &b) {
  return div(a, b);
}

template <class T> Tensor<T> &operator+=(Tensor<T> &a, const Tensor<T> &b) {
  a = add(a, b);
  return a;
}
template <class T> Tensor<T> &operator-=(Tensor<T> &a, const Tensor<T> &b) {
  a = sub(a, b);
  return a;
}
template <class T> Tensor<T> &operator*=(Tensor<T> &a, const Tensor<T> &b) {
  a = mul(a, b);
  return a;
}
template <class T> Tensor<T> &operator/=(Tensor<T> &a, const Tensor<T> &b) {
  a = div(a, b);
  return a;
}

// ------------------------------------------

template <class Value>
void unpack(const Tensor<Value> &tensor, std::vector<Var<Value>> &vector) {

  static Factory::Key<const TensorInfo *>::Value<const Operator *> factory{
      [](const TensorInfo *tensor_info) {
        std::vector<Operator::Argument> args;
        args.push_back(Operator::Argument::makeInput(tensor_info->type()));
        for (size_t i = 0; i < tensor_info->shape().elementCount(); i++) {
          args.push_back(
              Operator::Argument::makeOutput(TypeInfo::get<Value>()));
        }

        size_t element_count = tensor_info->shape().elementCount();
        const Operator *op = makeListOperator(
            std::string() + "unpack_" + args.front().typeInfo().name(),
            "unpack", args,
            [element_count](const ArgList &args) {
              TRACTOR_PROFILER("tensor unpack nl");
              for (size_t i = 0; i < element_count; i++) {
                args.arg<Value>(i + 1) = args.argp<Value>(0)[i];
              }
            },
            [element_count](const ArgList &args) {
              TRACTOR_PROFILER("tensor unpack f");
              size_t d = element_count + 1;
              for (size_t i = 0; i < element_count; i++) {
                args.arg<Value>(d + i + 1) = args.argp<Value>(d)[i];
              }
            },
            [element_count](const ArgList &args) {
              TRACTOR_PROFILER("tensor unpack r");
              size_t d = element_count + 1;
              for (size_t i = 0; i < element_count; i++) {
                args.argp<Value>(d)[i] = args.arg<Value>(d + i + 1);
              }
            });
        return op;
      }};

  auto *op = factory.get(tensor.info());

  vector.resize(tensor.info()->shape().elementCount());

  std::vector<void *> args;
  {
    args.push_back((void *)tensor.data());
    for (auto &e : vector) {
      args.push_back((void *)&e);
    }
  }

  callAndRecord(op, args.data());
}

template <class Value>
std::vector<Var<Value>> unpack(const Tensor<Value> &tensor) {
  std::vector<Var<Value>> ret;
  unpack(tensor, ret);
  return ret;
}

// ------------------------------------------

template <class Value>
Tensor<Value> pack_tensor(const Var<Value> *data, const TensorShape &shape) {

  static Factory::Key<const TensorInfo *>::Value<const Operator *> factory{
      [](const TensorInfo *tensor_info) {
        std::vector<Operator::Argument> args(
            tensor_info->shape().elementCount(),
            Operator::Argument::makeInput(TypeInfo::get<Value>()));
        args.push_back(Operator::Argument::makeOutput(tensor_info->type()));

        size_t element_count = tensor_info->shape().elementCount();
        const Operator *op = makeListOperator(
            std::string() + "pack_" + args.back().typeInfo().name(), "pack",
            args,
            [element_count](const ArgList &args) {
              TRACTOR_PROFILER("tensor pack nl");
              for (size_t i = 0; i < element_count; i++) {
                args.argp<Value>(element_count)[i] = args.arg<Value>(i);
              }
            },
            [element_count](const ArgList &args) {
              TRACTOR_PROFILER("tensor pack f");
              size_t d = element_count + 1;
              for (size_t i = 0; i < element_count; i++) {
                args.argp<Value>(d + element_count)[i] = args.arg<Value>(d + i);
              }
            },
            [element_count](const ArgList &args) {
              TRACTOR_PROFILER("tensor pack r");
              size_t d = element_count + 1;
              for (size_t i = 0; i < element_count; i++) {
                args.arg<Value>(d + i) = args.argp<Value>(d + element_count)[i];
              }
            });
        return op;
      }};

  Tensor<Value> ret(shape);

  auto *op = factory.get(ret.info());

  std::vector<void *> args;
  {
    size_t n = shape.elementCount();
    for (size_t i = 0; i < n; i++) {
      args.push_back((void *)(data + i));
    }
    args.push_back(ret.data());
  }

  callAndRecord(op, args.data());

  return ret;
}

template <class Value>
Tensor<Value> pack_tensor(const std::vector<Var<Value>> &vector,
                          const TensorShape &shape) {
  if (vector.size() != shape.elementCount()) {
    throw std::runtime_error("pack failed, shapes not compatible");
  }
  return pack_tensor(vector.data(), shape);
}

template <class Value>
Tensor<Value> pack_tensor(const std::vector<Var<Value>> &vector) {
  return pack_tensor(vector.data(), TensorShape(vector.size()));
}

template <class Value>
Tensor<Value> make_tensor(const TensorShape &shape, const Value &v) {
  std::vector<Var<Value>> vector(shape.elementCount(), Var<Value>(v));
  return pack_tensor(vector, shape);
}

// ------------------------------------------

template <class Activation, class Weight>
Tensor<Activation> matmul(const Tensor<Activation> &inputs,
                          const Tensor<Weight> &weights) {

  if (weights.shape().dimensions() != 2) {
    throw std::runtime_error(
        "matmul invalid number of dimensions for weight matrix");
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
    throw std::runtime_error("matmul invalid input shape for "
                             "activation vector or batch of vectors");
  }
  if (input_vector_neurons != input_neurons) {
    throw std::runtime_error("matmul input vector length does not "
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
          TRACTOR_DEBUG("tensor mul arg " << a.typeInfo().name());
        }

        std::string base_name = "matmul";
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

        static const bool check = false;

        const Operator *op = makePointerOp(
            base_name, variant_name, args,

            [batch_size, input_neurons, output_neurons,
             profiler_nonlinear](const Activation *a, const Weight *b,
                                 Activation *x) TRACTOR_FAST {
              ProfilerScope profiler_scope(*profiler_nonlinear);

              if (true) {
                auto ma = Eigen::Map<
                    const Eigen::Matrix<Activation, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>,
                    Eigen::Unaligned>(a, batch_size, input_neurons);
                auto mb = Eigen::Map<
                    const Eigen::Matrix<Weight, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>,
                    Eigen::Unaligned>(b, input_neurons, output_neurons);
                auto mx =
                    Eigen::Map<Eigen::Matrix<Activation, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>,
                               Eigen::Unaligned>(x, batch_size, output_neurons);
                mx.noalias() = ma * mb;
              }

              if (check) {
                for (size_t batch_index = 0; batch_index < batch_size;
                     batch_index++) {
                  for (size_t output_neuron = 0; output_neuron < output_neurons;
                       output_neuron++) {
                    Activation v = Activation(0);
                    for (size_t input_neuron = 0; input_neuron < input_neurons;
                         input_neuron++) {
                      v +=
                          a[batch_index * input_neurons + input_neuron] *
                          Activation(
                              b[input_neuron * output_neurons + output_neuron]);
                    }
                    TRACTOR_ASSERT(
                        std::abs(
                            x[batch_index * output_neurons + output_neuron] -
                            v) < 1e-9);
                  }
                }
              }
            },

            [batch_size, input_neurons, output_neurons,
             profiler_forward](const Activation *a, const Weight *b,
                               const Activation *x, const Activation *da,
                               const Weight *db, Activation *dx) TRACTOR_FAST {
              ProfilerScope profiler_scope(*profiler_forward);

              if (true) {
                // TRACTOR_INFO("pointer a " << a);
                // TRACTOR_INFO("pointer b " << b);
                // TRACTOR_INFO("pointer da " << da);
                // TRACTOR_INFO("pointer db " << db);
                // TRACTOR_INFO("pointer dx " << dx);
                {
                  auto ma = Eigen::Map<
                      const Eigen::Matrix<Activation, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                      Eigen::Unaligned>(a, batch_size, input_neurons);
                  auto mb = Eigen::Map<
                      const Eigen::Matrix<Weight, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                      Eigen::Unaligned>(b, input_neurons, output_neurons);
                  auto mda = Eigen::Map<
                      const Eigen::Matrix<Activation, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                      Eigen::Unaligned>(da, batch_size, input_neurons);
                  auto mdb = Eigen::Map<
                      const Eigen::Matrix<Weight, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                      Eigen::Unaligned>(db, input_neurons, output_neurons);
                  auto mdx =
                      Eigen::Map<Eigen::Matrix<Activation, Eigen::Dynamic,
                                               Eigen::Dynamic, Eigen::RowMajor>,
                                 Eigen::Unaligned>(dx, batch_size,
                                                   output_neurons);
                  // mdx.noalias() = (ma * mdb + mda * mb);
                  mdx.noalias() = ma * mdb;
                  mdx.noalias() += mda * mb;
                }
                if (check) {
                  bool error = false;
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
                      if (!(abs(dx[batch_index * output_neurons +
                                   output_neuron] -
                                dv) <= 1e-9)) {
                        error = true;
                      }
                    }
                  }
                  if (error) {
                    for (size_t batch_index = 0; batch_index < batch_size;
                         batch_index++) {
                      for (size_t output_neuron = 0;
                           output_neuron < output_neurons; output_neuron++) {
                        Activation dv = Activation(0);
                        for (size_t input_neuron = 0;
                             input_neuron < input_neurons; input_neuron++) {
                          dv +=
                              da[batch_index * input_neurons + input_neuron] *
                                  Activation(b[input_neuron * output_neurons +
                                               output_neuron]) +
                              a[batch_index * input_neurons + input_neuron] *
                                  Activation(db[input_neuron * output_neurons +
                                                output_neuron]);
                        }
                        std::cout
                            << dx[batch_index * output_neurons + output_neuron]
                            << "/" << dv << " ";
                      }
                      std::cout << "\n";
                    }
                    std::cout << "\n";
                    throw std::runtime_error("matmul error");
                  }
                }
              }

              if (false) {
                for (size_t batch_index = 0; batch_index < batch_size;
                     batch_index++) {
                  for (size_t output_neuron = 0; output_neuron < output_neurons;
                       output_neuron++) {
                    Activation dv = Activation(0);
                    for (size_t input_neuron = 0; input_neuron < input_neurons;
                         input_neuron++) {
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
              }
            },

            [batch_size, input_neurons, output_neurons, profiler_reverse](
                const Activation *a, const Weight *b, const Activation *x,
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
              }

              if (false) {
                {
                  TRACTOR_PROFILER("dense backprop left");
                  for (size_t batch_index = 0; batch_index < batch_size;
                       batch_index++) {
                    for (size_t input_neuron = 0; input_neuron < input_neurons;
                         input_neuron++) {
                      Activation dv = Activation(0);
                      for (size_t output_neuron = 0;
                           output_neuron < output_neurons; output_neuron++) {
                        dv += dx[batch_index * output_neurons + output_neuron] *
                              Activation(b[input_neuron * output_neurons +
                                           output_neuron]);
                      }
                      da[batch_index * input_neurons + input_neuron] = dv;
                    }
                  }
                }
                {
                  TRACTOR_PROFILER("dense backprop right");
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
                      db[input_neuron * output_neurons + output_neuron] = w;
                    }
                  }
                }
              }

              if (false) {
                {
                  TRACTOR_PROFILER("dense backprop left");
                  for (size_t batch_index = 0; batch_index < batch_size;
                       batch_index++) {
                    for (size_t input_neuron = 0; input_neuron < input_neurons;
                         input_neuron++) {
                      Activation dv = Activation(0);
                      for (size_t output_neuron = 0;
                           output_neuron < output_neurons; output_neuron++) {
                        dv += dx[batch_index * output_neurons + output_neuron] *
                              Activation(b[input_neuron * output_neurons +
                                           output_neuron]);
                      }
                      da[batch_index * input_neurons + input_neuron] = dv;
                    }
                  }
                }
                {
                  TRACTOR_PROFILER("dense backprop right");
                  for (size_t input_neuron = 0; input_neuron < input_neurons;
                       input_neuron++) {
                    for (size_t output_neuron = 0;
                         output_neuron < output_neurons; output_neuron++) {
                      Weight w = Weight(0);
                      for (size_t batch_index = 0; batch_index < batch_size;
                           batch_index++) {
                        w += batch_sum(
                            dx[batch_index * output_neurons + output_neuron] *
                            a[batch_index * input_neurons + input_neuron]);
                      }
                      db[input_neuron * output_neurons + output_neuron] = w;
                    }
                  }
                }
              }

              if (false) {
                // {
                //   TRACTOR_PROFILER("dense backprop left");
                //   for (size_t batch_index = 0; batch_index < batch_size;
                //        batch_index++) {
                //     for (size_t input_neuron = 0; input_neuron <
                //     input_neurons;
                //          input_neuron++) {
                //       Activation dv = Activation(0);
                //       for (size_t output_neuron = 0;
                //            output_neuron < output_neurons; output_neuron++) {
                //         dv += dx[batch_index * output_neurons +
                //         output_neuron] *
                //               Activation(b[input_neuron * output_neurons +
                //                            output_neuron]);
                //       }
                //       da[batch_index * input_neurons + input_neuron] = dv;
                //     }
                //   }
                // }

                {
                  TRACTOR_PROFILER("dense backprop left");
                  auto mb = Eigen::Map<
                      const Eigen::Matrix<Weight, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                      Eigen::Unaligned>(b, input_neurons, output_neurons);
                  auto mda =
                      Eigen::Map<Eigen::Matrix<Activation, Eigen::Dynamic,
                                               Eigen::Dynamic, Eigen::RowMajor>,
                                 Eigen::Unaligned>(da, batch_size,
                                                   input_neurons);
                  auto mdx = Eigen::Map<
                      const Eigen::Matrix<Activation, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                      Eigen::Unaligned>(dx, batch_size, output_neurons);
                  mda = mdx * mb.transpose();
                }

                // {
                //   TRACTOR_PROFILER("check dense backprop left");
                //   for (size_t batch_index = 0; batch_index < batch_size;
                //        batch_index++) {
                //     for (size_t input_neuron = 0; input_neuron <
                //     input_neurons;
                //          input_neuron++) {
                //       Activation dv = Activation(0);
                //       for (size_t output_neuron = 0;
                //            output_neuron < output_neurons; output_neuron++) {
                //         dv += dx[batch_index * output_neurons +
                //         output_neuron] *
                //               Activation(b[input_neuron * output_neurons +
                //                            output_neuron]);
                //       }
                //       TRACTOR_ASSERT(
                //           std::abs(
                //               da[batch_index * input_neurons + input_neuron]
                //               - dv) < 1e-9);
                //     }
                //   }
                // }

                {
                  TRACTOR_PROFILER("dense backprop zero");
                  std::memset(db, 0,
                              sizeof(Weight) * input_neurons * output_neurons);
                }

                // {
                //   TRACTOR_PROFILER("dense backprop right");
                //   for (size_t input_neuron = 0; input_neuron < input_neurons;
                //        input_neuron++) {
                //     for (size_t output_neuron = 0;
                //          output_neuron < output_neurons; output_neuron++) {
                //       for (size_t batch_index = 0; batch_index < batch_size;
                //            batch_index++) {
                //         db[input_neuron * output_neurons + output_neuron] +=
                //             batch_sum(
                //                 dx[batch_index * output_neurons +
                //                    output_neuron] *
                //                 a[batch_index * input_neurons +
                //                 input_neuron]);
                //       }
                //     }
                //   }
                // }

                // {
                //   TRACTOR_PROFILER("dense backprop right");
                //   for (size_t input_neuron = 0; input_neuron < input_neurons;
                //        input_neuron++) {
                //     for (size_t batch_index = 0; batch_index < batch_size;
                //          batch_index++) {
                //       for (size_t output_neuron = 0;
                //            output_neuron < output_neurons; output_neuron++) {
                //         db[input_neuron * output_neurons + output_neuron] +=
                //             batch_sum(
                //                 dx[batch_index * output_neurons +
                //                    output_neuron] *
                //                 a[batch_index * input_neurons +
                //                 input_neuron]);
                //       }
                //     }
                //   }
                // }

                {
                  TRACTOR_PROFILER("dense backprop right");
                  for (size_t batch_index = 0; batch_index < batch_size;
                       batch_index++) {
                    for (size_t input_neuron = 0; input_neuron < input_neurons;
                         input_neuron++) {
                      for (size_t output_neuron = 0;
                           output_neuron < output_neurons; output_neuron++) {
                        db[input_neuron * output_neurons + output_neuron] +=
                            batch_sum(
                                dx[batch_index * output_neurons +
                                   output_neuron] *
                                a[batch_index * input_neurons + input_neuron]);
                      }
                    }
                  }
                }
              }

              if (true) {
                {
                  // TRACTOR_PROFILER("dense backprop left");
                  auto mb = Eigen::Map<
                      const Eigen::Matrix<Weight, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                      Eigen::Unaligned>(b, input_neurons, output_neurons);
                  auto mda =
                      Eigen::Map<Eigen::Matrix<Activation, Eigen::Dynamic,
                                               Eigen::Dynamic, Eigen::RowMajor>,
                                 Eigen::Unaligned>(da, batch_size,
                                                   input_neurons);
                  auto mdx = Eigen::Map<
                      const Eigen::Matrix<Activation, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                      Eigen::Unaligned>(dx, batch_size, output_neurons);
                  mda.noalias() = mdx * mb.transpose();
                }
                if (check) {
                  TRACTOR_PROFILER("check dense backprop left");
                  for (size_t batch_index = 0; batch_index < batch_size;
                       batch_index++) {
                    for (size_t input_neuron = 0; input_neuron < input_neurons;
                         input_neuron++) {
                      Activation dv = Activation(0);
                      for (size_t output_neuron = 0;
                           output_neuron < output_neurons; output_neuron++) {
                        dv += dx[batch_index * output_neurons + output_neuron] *
                              Activation(b[input_neuron * output_neurons +
                                           output_neuron]);
                      }
                      TRACTOR_ASSERT(
                          std::abs(
                              da[batch_index * input_neurons + input_neuron] -
                              dv) < 1e-9);
                    }
                  }
                }
                {
                  // TRACTOR_PROFILER("dense backprop right");
                  auto ma = Eigen::Map<
                      const Eigen::Matrix<Activation, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                      Eigen::Unaligned>(a, batch_size, input_neurons);
                  auto mdb =
                      Eigen::Map<Eigen::Matrix<Weight, Eigen::Dynamic,
                                               Eigen::Dynamic, Eigen::RowMajor>,
                                 Eigen::Unaligned>(db, input_neurons,
                                                   output_neurons);
                  auto mdx = Eigen::Map<
                      const Eigen::Matrix<Activation, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                      Eigen::Unaligned>(dx, batch_size, output_neurons);
                  mdb.noalias() = ma.transpose() * mdx;
                }
                if (check) {
                  TRACTOR_PROFILER("check dense backprop right");
                  for (size_t input_neuron = 0; input_neuron < input_neurons;
                       input_neuron++) {
                    for (size_t output_neuron = 0;
                         output_neuron < output_neurons; output_neuron++) {
                      Weight w = Weight(0);
                      for (size_t batch_index = 0; batch_index < batch_size;
                           batch_index++) {
                        w += batch_sum(
                            dx[batch_index * output_neurons + output_neuron] *
                            a[batch_index * input_neurons + input_neuron]);
                      }
                      TRACTOR_ASSERT(std::abs(db[input_neuron * output_neurons +
                                                 output_neuron] -
                                              w) < 1e-9);
                    }
                  }
                }
              }

              if (false) {
                {
                  TRACTOR_PROFILER("dense backprop left");
                  for (size_t batch_index = 0; batch_index < batch_size;
                       batch_index++) {
                    for (size_t input_neuron = 0; input_neuron < input_neurons;
                         input_neuron++) {
                      Activation dv = Activation(0);
                      for (size_t output_neuron = 0;
                           output_neuron < output_neurons; output_neuron++) {
                        dv += dx[batch_index * output_neurons + output_neuron] *
                              Activation(b[input_neuron * output_neurons +
                                           output_neuron]);
                      }
                      da[batch_index * input_neurons + input_neuron] = dv;
                    }
                  }
                }
                {
                  TRACTOR_PROFILER("dense backprop right");
                  size_t input_neuron_output_neurons = 0;
                  for (size_t input_neuron = 0; input_neuron < input_neurons;
                       input_neuron++) {
                    for (size_t output_neuron = 0;
                         output_neuron < output_neurons; output_neuron++) {
                      Weight w = Weight(0);
                      size_t batch_index_input_neurons = 0;
                      size_t batch_index_output_neurons = 0;
                      for (size_t batch_index = 0; batch_index < batch_size;
                           batch_index++) {
                        Weight v = Weight(0);
                        batch_sum(
                            dx[batch_index_output_neurons + output_neuron] *
                                a[batch_index_input_neurons + input_neuron],
                            v);
                        w += v;
                        batch_index_input_neurons += input_neurons;
                        batch_index_output_neurons += output_neurons;
                      }
                      db[input_neuron_output_neurons + output_neuron] = w;
                    }
                    input_neuron_output_neurons += output_neurons;
                  }
                }
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

} // namespace tractor
