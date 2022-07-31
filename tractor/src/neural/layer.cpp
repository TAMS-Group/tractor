// (c) 2020-2022 Philipp Ruppel

#include <tractor/neural/layer.h>

#include <tractor/core/ops.h>
#include <tractor/core/program.h>
#include <tractor/neural/ops.h>
#include <tractor/tensor/ops.h>

#include <map>
#include <unordered_map>

namespace tractor {

// TODO: generate new type for tensor!;

template <class T>
Tensor<T> applyActivation(const Tensor<T> &input_tensor,
                          const Activation &activation) {
  Tensor<T> tensor = input_tensor;
  size_t tensor_size = tensor.size();
  switch (activation) {
  case Activation::Linear:
    break;
  case Activation::TanH:
    for (size_t i = 0; i < tensor_size; i++) {
      tensor[i] = tanh(tensor[i]);
    }
    break;
  case Activation::ReLU:
    for (size_t i = 0; i < tensor_size; i++) {
      tensor[i] = relu(tensor[i]);
    }
    break;
  default:
    throw std::runtime_error("unsupported activation");
    break;
  }
  return tensor;
}

template Tensor<Var<Batch<double, 4>>>
applyActivation(const Tensor<Var<Batch<double, 4>>> &input_tensor,
                const Activation &activation);

OpGroup makeOpGroup(const std::string &name) {
  static std::unordered_map<std::string, std::shared_ptr<int>> map;
  if (map.find(name) == map.end()) {
    map[name] = std::make_shared<int>(1);
  }
  return OpGroup(map[name].get());
}

class ListOperator : public Operator {

public:
  ListOperator(const std::string &name, const OpMode &mode,
               const OpGroup &group, const std::vector<Argument> &args,
               void (*callback)(const void *base, const uintptr_t *offsets))
      : Operator(name, mode, this, group) {
    _arguments = args;
    _argument_count = args.size();
    _functions.indirect = callback;
  }
};

const Operator *makeListOperator(const std::string &name, const OpMode &mode,
                                 const OpGroup &group,
                                 const std::vector<Operator::Argument> &args,
                                 void (*callback)(const void *base,
                                                  const uintptr_t *offsets)) {
  static std::unordered_map<std::string, std::shared_ptr<Operator>> map;
  if (map.find(name) == map.end()) {
    map[name] =
        std::make_shared<ListOperator>(name, mode, group, args, callback);
  }
  return map[name].get();
}

// struct TensorInfo {
//   std::vector<Var<uint64_t>> shape;
//   size_t size = 0;
//   size_t stride = 0;
//   uintptr_t data = 0;
// };
// template <class Scalar>
// void recordTensorArg(std::vector<TensorInfo> &infos,
//                      const Tensor<Scalar> &arg) {
//   TensorInfo info;
//   info.shape.emplace_back(Var<uint64_t>(arg.shape().size()));
//   for (auto &s : arg.shape()) {
//     info.shape.emplace_back(Var<uint64_t>(s));
//   }
//   info.size = arg.size();
//   info.stride = sizeof(Scalar);
//   info.data = (uintptr_t)arg.data();
//   infos.push_back(info);
// }
// template <class Scalar>
// void recordTensorArgs(std::vector<TensorInfo> &infos,
//                       const Tensor<Scalar> &arg) {
//   recordTensorArg(infos, arg);
// }
// template <class Scalar, class... Args>
// void recordTensorArgs(std::vector<TensorInfo> &infos, const Tensor<Scalar>
// &arg,
//                       Args &&...args) {
//   recordTensorArg(infos, arg);
//   recordTensorArgs(infos, args...);
// }

// template <class Scalar>
// void buildTensorArgs(std::vector<Operator::Argument> &list,
//                      const Tensor<Scalar> &tensor) {
//   list.push_back(Operator::Argument::make<const uint64_t &>());
//   for (size_t i = 0; i < tensor.shape().size(); i++) {
//     list.push_back(Operator::Argument::make<const uint64_t &>());
//   }
//   for (size_t i = 0; i < tensor.size(); i++) {
//     list.push_back(Operator::Argument::make<decltype(value(tensor[0]))>());
//   }
// }
// template <class Scalar>
// void buildTensorArgs(std::vector<Operator::Argument> &list,
//                      Tensor<Scalar> &tensor) {
//   list.push_back(Operator::Argument::make<uint64_t &>());
//   for (size_t i = 0; i < tensor.shape().size(); i++) {
//     list.push_back(Operator::Argument::make<uint64_t &>());
//   }
//   for (size_t i = 0; i < tensor.size(); i++) {
//     list.push_back(Operator::Argument::make<decltype(value(tensor[0]))>());
//   }
// }
// template <class Tensor, class... Args>
// void buildTensorArgs(std::vector<Operator::Argument> &list, Tensor &&tensor,
//                      Args &&...args) {
//   buildTensorArgs(list, tensor);
//   buildTensorArgs(list, args...);
// }
// template <class... Args>
// std::vector<Operator::Argument> makeTensorArgs(Args &&...args) {
//   std::vector<Operator::Argument> ret;
//   buildTensorArgs(ret, args...);
//   return ret;
// }

template <class Args>
std::vector<Operator::Argument> makeForwardArgs(const Args &args) {
  std::vector<Operator::Argument> ret;
  for (auto &a : args) {
    ret.push_back(a.makeInput());
  }
  for (auto &a : args) {
    ret.push_back(a);
  }
  return ret;
}

template <class Args>
std::vector<Operator::Argument> makeReverseArgs(const Args &args) {
  std::vector<Operator::Argument> ret;
  for (auto &a : args) {
    ret.push_back(a.makeInput());
  }
  for (auto &a : args) {
    ret.push_back(a.makeReverse());
  }
  return ret;
}

const Operator *makeListOperator(
    const std::string &name, const std::vector<Operator::Argument> &args,
    void (*fun_compute)(const void *base, const uintptr_t *offsets),
    void (*fun_forward)(const void *base, const uintptr_t *offsets),
    void (*fun_reverse)(const void *base, const uintptr_t *offsets)) {
  auto group = makeOpGroup(name);
  auto *ret = makeListOperator(name, OpMode(typeid(compute *)), group, args,
                               fun_compute);
  makeListOperator("forward_" + name, OpMode(typeid(forward *)), group,
                   makeForwardArgs(args), fun_forward);
  makeListOperator("reverse_" + name, OpMode(typeid(reverse *)), group,
                   makeReverseArgs(args), fun_reverse);
  return ret;
}

class ArgList {
  const void *_base = nullptr;
  const uintptr_t *_offsets = nullptr;

public:
  ArgList(const void *base, const uintptr_t *offsets)
      : _base(base), _offsets(offsets) {}
  template <class T> T &arg(size_t i) const {
    return *(T *)(void *)((uint8_t *)_base + _offsets[i]);
  }
};

// template <class T> class TensorArg : public ArgList {
//   size_t _size = 0;
//   template <class... Indices> size_t _index(const Indices &...indices) const
//   {
//     std::array<size_t, sizeof...(Indices)> ii = {indices...};
//     size_t ret = 0;
//     for (size_t i = 0; i < sizeof...(Indices); i++) {
//       ret *= shape(i);
//       ret += ii[i];
//     }
//     return ret;
//   }
//
// public:
//   TensorArg(const void *base, const uintptr_t *offsets)
//       : ArgList(base, offsets) {
//     size_t dims = dimensions();
//     if (dims > 0) {
//       _size = 1;
//       for (size_t i = 0; i < dims; i++) {
//         _size *= shape(i);
//       }
//     }
//   }
//   size_t dimensions() const { return arg<uint64_t>(0); }
//   size_t shape(size_t i) const { return arg<uint64_t>(i + 1); }
//   size_t size() const { return _size; }
//   template <class... Indices>
//   auto &operator()(const Indices &...indices) const {
//     return arg<T>(1 + dimensions() + _index(indices...));
//   }
//   template <class... Indices> auto &operator()(const Indices &...indices) {
//     return arg<T>(1 + dimensions() + _index(indices...));
//   }
// };
//
// template <class T>
// TensorArg<T> bindTensorArg(const void *base, const uintptr_t **offsets) {
//   TensorArg<T> ret(base, *offsets);
//   (*offsets) += (1 + ret.dimensions() + ret.size());
//   return ret;
// }

template <class T, size_t Dimensions> class TensorArg : public ArgList {
  std::array<size_t, Dimensions> _shape;
  template <class... Indices> size_t _index(const Indices &...indices) const {
    if (_shape.size() != sizeof...(Indices)) {
      throw std::runtime_error("incorrect number of tensor index dimensions");
    }
    std::array<size_t, sizeof...(Indices)> ii = {indices...};
    size_t ret = 0;
    for (size_t i = 0; i < sizeof...(Indices); i++) {
      if (ii.at(i) >= _shape.at(i)) {
        throw std::runtime_error("tensor index out of range");
      }
      ret *= _shape[i];
      ret += ii[i];
    }
    return ret;
  }

public:
  TensorArg(const void *base, const uintptr_t *offsets,
            const std::array<size_t, Dimensions> &shape)
      : ArgList(base, offsets), _shape(shape) {}
  size_t dimensions() const { return _shape.size(); }
  const std::array<size_t, Dimensions> &shape() const { return _shape; }
  size_t size() const {
    if (_shape.empty()) {
      return 0;
    }
    size_t s = 1;
    for (auto &v : _shape) {
      s *= v;
    }
    return s;
  }
  template <class... Indices>
  auto &operator()(const Indices &...indices) const {
    return arg<T>(_index(indices...));
  }
};

template <class T, size_t N>
TensorArg<T, N> bindTensorArg(const void *base, const uintptr_t **offsets,
                              const std::array<size_t, N> &shape) {
  TensorArg<T, N> ret(base, *offsets, shape);
  (*offsets) += ret.size();
  return ret;
}

template <class T> T &bindArg(const void *base, const uintptr_t **offsets) {
  T *ret = (T *)(void *)((uint8_t *)(void *)base + (*offsets)[0]);
  (*offsets)++;
  return *ret;
}

template <class T, size_t N>
std::ostream &operator<<(std::ostream &stream, const TensorArg<T, N> &arg) {
  stream << "tensor[" << arg.dimensions();
  for (size_t i = 0; i < arg.dimensions(); i++) {
    stream << "," << arg.shape()[i];
  }
  stream << "]";
  return stream;
}

template <class T> T batchSum(const T &v) { return v; }
template <class T, size_t N> T batchSum(const Batch<T, N> &v) {
  T ret = T(0);
  for (size_t i = 0; i < N; i++) {
    ret += v[i];
  }
  return ret;
}

template <class ActivationScalar, class WeightScalar>
Tensor<ActivationScalar> dense_mul_vec_mat(const Tensor<ActivationScalar> &a,
                                           const Tensor<WeightScalar> &b) {

  uint64_t rows = b.shape()[0];
  uint64_t cols = b.shape()[1];

  Tensor<ActivationScalar> r;
  r.resize(cols);

  typedef typename std::decay<decltype(value(a[0]))>::type Batch;
  typedef typename std::decay<decltype(value(b[0]))>::type Weight;

  if (auto rec = Recorder::instance()) {

    std::cout << "record tensor dense op " << rows << " " << cols << std::endl;

    std::string name = "dense_mul_vec_mat_" + std::to_string(rows) + "_" +
                       std::to_string(cols) + "_" +
                       typeid(ActivationScalar).name();

    std::vector<Operator::Argument> arguments;
    arguments.push_back(Operator::Argument::make<const uint64_t &>());
    arguments.push_back(Operator::Argument::make<const uint64_t &>());
    for (size_t row = 0; row < rows; row++) {
      arguments.push_back(Operator::Argument::make<const Batch &>());
    }
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        arguments.push_back(Operator::Argument::make<const Weight &>());
      }
    }
    for (size_t col = 0; col < cols; col++) {
      arguments.push_back(Operator::Argument::make<Batch &>());
    }
    std::cout << "args " << arguments.size() << std::endl;

    // throw 0;

    const Operator *op = makeListOperator(
        name,
        // makeTensorArgs(a, b, r),
        arguments,
        [](const void *base, const uintptr_t *offsets) {
          auto rows = bindArg<uint64_t>(base, &offsets);
          auto cols = bindArg<uint64_t>(base, &offsets);
          auto a = bindTensorArg<Batch, 1>(base, &offsets, {rows});
          auto b = bindTensorArg<Weight, 2>(base, &offsets, {rows, cols});
          auto x = bindTensorArg<Batch, 1>(base, &offsets, {cols});
          // std::cout << firstBatchElement(a(0)) << " " << b(0, 0) << " "
          //           << firstBatchElement(x(0)) << std::endl;
          // std::cout << "op dense " << rows << " " << cols << " " << a << " "
          //           << b << " " << r << std::endl;
          for (size_t col = 0; col < cols; col++) {
            Batch s = Batch(0);
            for (size_t row = 0; row < rows; row++) {
              // std::cout << row << " " << col << " " << b(row, col) <<
              // std::endl;
              s += a(row) * Batch(b(row, col));
            }
            x(col) = s;
          }
        },
        [](const void *base, const uintptr_t *offsets) {
          auto rows = bindArg<uint64_t>(base, &offsets);
          auto cols = bindArg<uint64_t>(base, &offsets);
          auto a = bindTensorArg<Batch, 1>(base, &offsets, {rows});
          auto b = bindTensorArg<Weight, 2>(base, &offsets, {rows, cols});
          auto x = bindTensorArg<Batch, 1>(base, &offsets, {cols});
          auto drows = bindArg<uint64_t>(base, &offsets);
          auto dcols = bindArg<uint64_t>(base, &offsets);
          auto da = bindTensorArg<Batch, 1>(base, &offsets, {rows});
          auto db = bindTensorArg<Weight, 2>(base, &offsets, {rows, cols});
          auto dx = bindTensorArg<Batch, 1>(base, &offsets, {cols});
          // std::cout << "op dense forward" << rows << " " << cols << " " << a
          //           << " " << b << " " << r << " " << da << " " << db << " "
          //           << dr << std::endl;
          // for (size_t row = 0; row < rows; row++) {
          //   std::cout << row << " " << firstBatchElement(a(row)) << " "
          //             << firstBatchElement(da(row)) << std::endl;
          // }
          // throw 0;
          for (size_t col = 0; col < cols; col++) {
            Batch s = Batch(0);
            for (size_t row = 0; row < rows; row++) {
              // std::cout << row << " " << col << " " << b(row, col) << " "
              //           << db(row, col) << std::endl;
              s += da(row) * Batch(b(row, col)) + a(row) * Batch(db(row, col));
            }
            dx(col) = s;
          }
        },
        [](const void *base, const uintptr_t *offsets) {
          auto rows = bindArg<uint64_t>(base, &offsets);
          auto cols = bindArg<uint64_t>(base, &offsets);
          auto a = bindTensorArg<Batch, 1>(base, &offsets, {rows});
          auto b = bindTensorArg<Weight, 2>(base, &offsets, {rows, cols});
          auto x = bindTensorArg<Batch, 1>(base, &offsets, {cols});
          auto drows = bindArg<uint64_t>(base, &offsets);
          auto dcols = bindArg<uint64_t>(base, &offsets);
          auto da = bindTensorArg<Batch, 1>(base, &offsets, {rows});
          auto db = bindTensorArg<Weight, 2>(base, &offsets, {rows, cols});
          auto dx = bindTensorArg<Batch, 1>(base, &offsets, {cols});
          // std::cout << "op dense reverse" << rows << " " << cols << " " << a
          //           << " " << b << " " << r << " " << da << " " << db << " "
          //           << dr << std::endl;
          for (size_t row = 0; row < rows; row++) {
            Batch s = Batch(0);
            for (size_t col = 0; col < cols; col++) {
              db(row, col) = batchSum(a(row) * dx(col));
              s += dx(col) * Batch(b(row, col));
            }
            da(row) = s;
          }
        });

    // const Operator *opf = op->variant<reverse>();
    // for (auto arg : opf->arguments()) {
    //   std::cout << arg.type().name() << " " << arg.isInput() << " "
    //             << arg.isOutput() << std::endl;
    // }
    // throw 10;

    Var<uint64_t> vrows(rows);
    Var<uint64_t> vcols(cols);
    rec->op(op);
    rec->arg(&vrows);
    rec->arg(&vcols);
    for (size_t row = 0; row < rows; row++) {
      rec->arg(&a(row));
    }
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        rec->arg(&b(row, col));
      }
    }
    for (size_t col = 0; col < cols; col++) {
      rec->arg(&r(col));
    }

    // std::vector<TensorInfo> infos;
    // recordTensorArgs(infos, a, b, r);
    // rec->op(op);
    // for (auto &info : infos) {
    //   for (auto &s : info.shape) {
    //     rec->arg(&s);
    //   }
    //   for (size_t i = 0; i < info.size; i++) {
    //     rec->push(info.data + i * info.stride);
    //   }
    // }
  }

  for (size_t col = 0; col < cols; col++) {
    Batch s = Batch(0);
    for (size_t row = 0; row < rows; row++) {
      s += value(a(row)) * Batch(value(b(row, col)));
    }
    value(r(col)) = s;
  }

  // throw 10;

  return r;
}

// template <class T> class TensorRef {
//   void *_base = nullptr;
//   uintptr_t *_offsets = nullptr;
//
// public:
//   TensorRef(const void *base, const uintptr_t *offsets)
//       : _base(base), _offsets(offsets) {}
// };
//
// template <class ActivationScalar, class WeightScalar> struct TensorOpDense {
//   struct Compute {
//     static void run(const Tensor<ActivationScalar> &a,
//                     const Tensor<WeightScalar> &b,
//                     Tensor<ActivationScalar> &x) {}
//   };
//   struct Forward {
//     static void forward(const Tensor<ActivationScalar> &a,
//                         const Tensor<WeightScalar> &b,
//                         const Tensor<ActivationScalar> &x,
//                         const Tensor<ActivationScalar> &da,
//                         const Tensor<WeightScalar> &db,
//                         Tensor<ActivationScalar> &dx) {}
//   };
//   struct Reverse {
//     static void reverse(const Tensor<ActivationScalar> &a,
//                         const Tensor<WeightScalar> &b,
//                         const Tensor<ActivationScalar> &x,
//                         Tensor<ActivationScalar> &da, Tensor<WeightScalar>
//                         &db, const Tensor<ActivationScalar> &dx) {}
//   };
// };
//
// struct TensorInfo {
//   std::vector<Var<uint64_t>> shape;
//   size_t size = 0;
//   size_t stride = 0;
//   uintptr_t data = 0;
// };
// template <class Scalar>
// void recordTensorArg(std::vector<TensorInfo> &infos,
//                      const Tensor<Scalar> &arg) {
//   // if (auto rec = Recorder::instance()) {
//   //   std::cout << "record tensor arg " << typeid(Scalar).name() <<
//   //   std::endl;
//   //   {
//   //     std::cout << "record tensor arg d " << arg.shape().size() <<
//   //     std::endl; Var<uint64_t> v((uint64_t)arg.shape().size());
//   //     rec->arg(&v);
//   //   }
//   //   for (size_t s : arg.shape()) {
//   //     std::cout << "record tensor arg s " << s << std::endl;
//   //     Var<uint64_t> v((uint64_t)s);
//   //     rec->arg(&v);
//   //   }
//   //   std::cout << "record tensor arg n " << arg.size() << std::endl;
//   //   for (size_t i = 0; i < arg.size(); i++) {
//   //     rec->arg(&(arg[i]));
//   //   }
//   // }
//   TensorInfo info;
//   info.shape.emplace_back(Var<uint64_t>(arg.shape().size()));
//   for (auto &s : arg.shape()) {
//     info.shape.emplace_back(Var<uint64_t>(s));
//   }
//   info.size = arg.size();
//   info.stride = sizeof(Scalar);
//   info.data = (uintptr_t)arg.data();
//   infos.push_back(info);
// }
// template <class Scalar>
// void recordTensorArgs(std::vector<TensorInfo> &infos,
//                       const Tensor<Scalar> &arg) {
//   recordTensorArg(infos, arg);
// }
// template <class Scalar, class... Args>
// void recordTensorArgs(std::vector<TensorInfo> &infos, const Tensor<Scalar>
// &arg,
//                       Args &&...args) {
//   recordTensorArg(infos, arg);
//   recordTensorArgs(infos, args...);
// }
//
// template <class Scalar>
// std::string makeTensorShapeName(const Tensor<Scalar> &tensor) {
//   std::string ret = "";
//   for (auto &v : tensor.shape()) {
//     ret += "_" + std::to_string(v);
//   }
//   return ret;
// }
// template <class Scalar, class... Args>
// std::string makeTensorShapeName(const Tensor<Scalar> &tensor, Args &&...args)
// {
//   return makeTensorShapeName(args...) + "_" + makeTensorShapeName(tensor);
// }
//
// struct TensorOperatorImplBase : Operator {
//   template <class Scalar> void makeTensorArgs(const Tensor<Scalar> &tensor) {
//     _arguments.push_back(Argument::make<const uint64_t &>());
//     for (size_t i = 0; i < tensor.shape().size(); i++) {
//       _arguments.push_back(Argument::make<const uint64_t &>());
//     }
//     for (size_t i = 0; i < tensor.size(); i++) {
//       _arguments.push_back(Argument::make<decltype(value(tensor[0]))>());
//     }
//   }
//   template <class Scalar> void makeTensorArgs(Tensor<Scalar> &tensor) {
//     _arguments.push_back(Argument::make<uint64_t &>());
//     for (size_t i = 0; i < tensor.shape().size(); i++) {
//       _arguments.push_back(Argument::make<uint64_t &>());
//     }
//     for (size_t i = 0; i < tensor.size(); i++) {
//       _arguments.push_back(Argument::make<decltype(value(tensor[0]))>());
//     }
//   }
//   template <class Tensor, class... Args>
//   void makeTensorArgs(Tensor &&tensor, Args &&...args) {
//     makeTensorArgs(tensor);
//     makeTensorArgs(args...);
//   }
//   TensorOperatorImplBase(const std::string &name, const OpMode &mode,
//                          const OpType &op, const OpGroup &group)
//       : Operator(name, mode, op, group) {}
// };
//
// template <class Op, class... Args>
// struct TensorOperatorImpl : TensorOperatorImplBase {
//   template<class Arg, class... Args>
//   static void _impl(const void *base, const uintptr_t *offsets) {
//
//   }
//   TensorOperatorImpl(const std::string &name, const OpMode &mode,
//                      const OpType &op, const OpGroup &group, Args &&...args)
//       : TensorOperatorImplBase(name, mode, op, group) {
//     makeTensorArgs(args...);
//     _argument_count = _arguments.size();
//     _functions.indirect = [](const void *base, const uintptr_t *offsets) {
//       std::cout << "TODO" << std::endl;
//     };
//   }
// };
//
// template <class Group, class Op, class... Args>
// const Operator *makeTensorOp(const std::string &base_name, const OpMode
// &mode,
//                              Args &&...args) {
//   std::string name = base_name + "_" + makeTensorShapeName(args...);
//   std::cout << "tensor op name " << name << std::endl;
//   static std::unordered_map<std::string, const Operator *> registry;
//   if (registry.find(name) == registry.end()) {
//     registry[name] = new TensorOperatorImpl<Op, Args...>(
//         name, mode, OpType(typeid(Op *)), OpGroup(typeid(Group *)), args...);
//   }
//   return registry[name];
// }
//
// template <class Op> struct TensorOperationRecorder {
//   template <class... Args>
//   static void record(const std::string &name, Args &&...args) {
//     if (auto rec = Recorder::instance()) {
//
//       std::vector<TensorInfo> infos;
//       recordTensorArgs(infos, args...);
//
//       auto *op = makeTensorOp<Op, typename Op::Compute>(
//           name, OpMode(typeid(compute *)), args...);
//
//       makeTensorOp<Op, typename Op::Forward>(
//           "forward_" + name, OpMode(typeid(forward *)), args..., args...);
//
//       makeTensorOp<Op, typename Op::Reverse>(
//           "reverse_" + name, OpMode(typeid(reverse *)), args..., args...);
//
//       rec->op(op);
//       // for (auto &arg : op->arguments()) {
//       //   std::cout << arg.type().name() << " " << (int)arg.isInput() << " "
//       //             << (int)arg.isOutput() << std::endl;
//       // }
//
//       size_t recorded_instructions = rec->instructions().size();
//
//       for (auto &info : infos) {
//         for (auto &s : info.shape) {
//           rec->arg(&s);
//         }
//         for (size_t i = 0; i < info.size; i++) {
//           rec->push(info.data + i * info.stride);
//         }
//       }
//
//       //   std::cout << "record tensor arg " << typeid(Scalar).name() <<
//       //   std::endl;
//       //   {
//       //     std::cout << "record tensor arg d " << arg.shape().size() <<
//       //     std::endl; Var<uint64_t> v((uint64_t)arg.shape().size());
//       //     rec->arg(&v);
//       //   }
//       //   for (size_t s : arg.shape()) {
//       //     std::cout << "record tensor arg s " << s << std::endl;
//       //     Var<uint64_t> v((uint64_t)s);
//       //     rec->arg(&v);
//       //   }
//       //   std::cout << "record tensor arg n " << arg.size() << std::endl;
//       //   for (size_t i = 0; i < arg.size(); i++) {
//       //     rec->arg(&(arg[i]));
//       //   }
//
//       recorded_instructions =
//           rec->instructions().size() - recorded_instructions;
//       if (recorded_instructions != op->argumentCount()) {
//         std::cout << recorded_instructions << " " << op->argumentCount()
//                   << std::endl;
//         throw std::runtime_error("argument count mismatch");
//       }
//     }
//   }
// };
//
// template <class ActivationScalar, class WeightScalar>
// Tensor<ActivationScalar> dense_mul_vec_mat(const Tensor<ActivationScalar> &a,
//                                            const Tensor<WeightScalar> &b) {
//   std::cout << "record tensor op" << std::endl;
//   Tensor<ActivationScalar> r;
//   r.resize(b.shape()[1]);
//   TensorOperationRecorder<
//       TensorOpDense<ActivationScalar, WeightScalar>>::record("dense", a, b,
//       r);
//   return r;
// }

// class op_dense;
//
// struct TensorOperatorBase : Operator {
//   TensorOperatorBase(const std::string &name, const OpMode &mode,
//                      const OpGroup &group)
//       : Operator(name, mode, this, group) {}
//   template <class T> void makeArgs(size_t count) {
//     for (size_t i = 0; i < count; i++) {
//       _arguments.push_back(Argument::make<T>());
//     }
//     _argument_count += count;
//   }
//   void makeFunctions(void (*callback)(const void *base,
//                                       const uintptr_t *offsets)) {
//     _functions.indirect = callback;
//   }
// };
//
// template <class ActivationScalar, class WeightScalar>
// struct OpDense : TensorOperatorBase {
//
//   static const Operator *instance(size_t rows, size_t cols) {
//     auto key = std::make_pair(rows, cols);
//     static std::map<std::pair<size_t, size_t>, OpDense *> map;
//     if (map.find(key) == map.end()) {
//       map[key] = new OpDense(rows, cols);
//     }
//     return map[key];
//   }
//
//   static std::string makeName(const std::string &prefix, size_t rows,
//                               size_t cols) {
//     return std::string() + prefix + "dense_" + std::to_string(rows) + "_" +
//            std::to_string(cols);
//   }
//
//   struct OpDenseForward : TensorOperatorBase {
//     OpDenseForward(size_t rows, size_t cols)
//         : TensorOperatorBase(makeName("forward_", rows, cols),
//                              typeid(forward *), typeid(op_dense *)) {
//
//       makeArgs<const uint64_t &>(2);
//       makeArgs<const ActivationScalar &>(rows);
//       makeArgs<const WeightScalar &>(rows * cols);
//       makeArgs<const ActivationScalar &>(cols);
//
//       makeArgs<const uint64_t &>(2);
//       makeArgs<const ActivationScalar &>(rows);
//       makeArgs<const WeightScalar &>(rows * cols);
//       makeArgs<ActivationScalar &>(cols);
//
//       makeFunctions([](const void *base, const uintptr_t *offsets) {
//         throw std::runtime_error("dense");
//         auto b = (const uint8_t *)base;
//         size_t rows = *(const uint64_t *)(b + offsets[0]);
//         size_t cols = *(const uint64_t *)(b + offsets[1]);
//         std::cout << "op dense" << std::endl;
//       });
//     }
//   } _forward;
//
//   struct OpDenseReverse : TensorOperatorBase {
//     OpDenseReverse(size_t rows, size_t cols)
//         : TensorOperatorBase(makeName("reverse_", rows, cols),
//                              typeid(reverse *), typeid(op_dense *)) {
//
//       makeArgs<const uint64_t &>(2);
//       makeArgs<const ActivationScalar &>(rows);
//       makeArgs<const WeightScalar &>(rows * cols);
//       makeArgs<ActivationScalar &>(cols);
//
//       makeArgs<uint64_t &>(2);
//       makeArgs<ActivationScalar &>(rows);
//       makeArgs<WeightScalar &>(rows * cols);
//       makeArgs<const ActivationScalar &>(cols);
//
//       makeFunctions([](const void *base, const uintptr_t *offsets) {
//         throw std::runtime_error("dense");
//         auto b = (const uint8_t *)base;
//         size_t rows = *(const uint64_t *)(b + offsets[0]);
//         size_t cols = *(const uint64_t *)(b + offsets[1]);
//         std::cout << "op dense" << std::endl;
//       });
//     }
//   } _reverse;
//
//   OpDense(size_t rows, size_t cols)
//       : TensorOperatorBase(makeName("", rows, cols), typeid(compute *),
//                            typeid(op_dense *)),
//         _forward(rows, cols), _reverse(rows, cols) {
//
//     makeArgs<const uint64_t &>(2);
//     makeArgs<const ActivationScalar &>(rows);
//     makeArgs<const WeightScalar &>(rows * cols);
//     makeArgs<ActivationScalar &>(cols);
//
//     makeFunctions([](const void *base, const uintptr_t *offsets) {
//       throw std::runtime_error("dense");
//       auto b = (const uint8_t *)base;
//       size_t rows = *(const uint64_t *)(b + offsets[0]);
//       size_t cols = *(const uint64_t *)(b + offsets[1]);
//       std::cout << "op dense" << std::endl;
//     });
//   }
// };
//
// template <class ActivationScalar, class WeightScalar>
// Tensor<ActivationScalar> dense_mul_vec_mat(const Tensor<ActivationScalar>
// &a,
//                                            const Tensor<WeightScalar> &b) {
//   TRACTOR_CHECK_TENSOR_DIMENSIONS(a, 1);
//   TRACTOR_CHECK_TENSOR_DIMENSIONS(b, 2);
//   if (a.size() != b.rows()) {
//     throw std::runtime_error("incompatible tensor shapes");
//   }
//
//   uint64_t rows = b.shape()[0];
//   uint64_t cols = b.shape()[1];
//
//   Tensor<ActivationScalar> r;
//   r.resize(cols);
//
//   Var<uint64_t> vrows(rows);
//   Var<uint64_t> vcols(cols);
//
//   auto rec = Recorder::instance();
//   // std::cout << "rec " << rows << " " << cols << " "
//   //           << typeid(ActivationScalar).name() << " " << std::endl;
//   // if (rec)
//   //   throw std::runtime_error("rec");
//   if (rec) {
//     const Operator *op = OpDense<
//         typename std::decay<decltype(value(a[0]))>::type,
//         typename std::decay<decltype(value(b[0]))>::type>::instance(rows,
//         cols);
//     std::cout << op << " " << typeid(*op).name() << std::endl;
//     for (auto &a : op->arguments()) {
//       std::cout << a.type().name() << " " << (int)a.isInput() << " "
//                 << (int)a.isOutput() << std::endl;
//     }
//     // throw std::runtime_error("op");
//     rec->op(op);
//     rec->arg(&vrows);
//     rec->arg(&vcols);
//     for (size_t row = 0; row < rows; row++) {
//       rec->arg(&a(row));
//     }
//     for (size_t row = 0; row < rows; row++) {
//       for (size_t col = 0; col < cols; col++) {
//         rec->arg(&b(row, col));
//       }
//     }
//     for (size_t col = 0; col < cols; col++) {
//       rec->arg(&r(col));
//     }
//   }
//
//   for (size_t row = 0; row < rows; row++) {
//     for (size_t col = 0; col < cols; col++) {
//       value(r(col)) =
//           value(a(row)) * decltype(value(a(row)))(value(b(row, col))) +
//           value(r(col));
//     }
//   }
//
//   return r;
// }

// template <class ActivationScalar, class WeightScalar>
// Tensor<ActivationScalar> dense_mul_vec_mat(const Tensor<ActivationScalar>
// &a,
//                                            const Tensor<WeightScalar> &b) {
//   TRACTOR_CHECK_TENSOR_DIMENSIONS(a, 1);
//   TRACTOR_CHECK_TENSOR_DIMENSIONS(b, 2);
//   if (a.size() != b.rows()) {
//     throw std::runtime_error("incompatible tensor shapes");
//   }
//   Tensor<ActivationScalar> r;
//   r.resize(b.shape()[1]);
//   size_t cols = b.shape()[1];
//   size_t rows = b.shape()[0];
//   for (size_t col = 0; col < cols; col++) {
//     size_t row = 0;
//     for (; row + 3 < rows; row += 4) {
//
//       r(col) = dense4(
//
//           a(row + 0), a(row + 1), a(row + 2), a(row + 3),
//
//           b(row + 0, col), b(row + 1, col), b(row + 2, col), b(row + 3,
//           col),
//
//           r(col));
//     }
//     for (; row < rows; row++) {
//       ActivationScalar weight;
//       batch(b(row, col), weight);
//       r(col) = madd(a(row), weight, r(col));
//     }
//   }
//   return r;
// }

template <class Scalar>
Tensor<Scalar>
DenseLayer<Scalar>::evaluate(const std::vector<Tensor<Scalar>> &inputs,
                             const LayerMode &mode) {
  auto &input = inputs.at(0);
  TRACTOR_CHECK_TENSOR_DIMENSIONS(input, 1);

  if (!_initialized) {
    _initialized = true;
    std::cout << "build dense layer " << input.size() << " x " << _units
              << std::endl;

    _weights.resize(input.size(), _units);
    randomize(_weights, _stdev);
    variable(_weights);

    if (_use_bias) {
      _bias.resize(_units);
      randomize(_bias, _stdev);
      variable(_bias);
    }

    if (_weight_regularization != 0) {
      for (size_t row = 0; row < input.size(); row++) {
        for (size_t col = 0; col < _units; col++) {
          goal(_weights(row, col) * _weight_regularization);
        }
      }
    }

    if (_use_bias) {
      if (_bias_regularization != 0) {
        for (size_t i = 0; i < _units; i++) {
          goal(_bias(i) * _bias_regularization);
        }
      }
    }
  }

  Tensor<Scalar> activity = dense_mul_vec_mat(input, _weights);

  if (_use_bias) {
    for (size_t i = 0; i < _units; i++) {
      Scalar bias;
      batch(_bias[i], bias);
      activity[i] += bias;
    }
  }

  if (_activity_regularization != 0) {
    for (size_t i = 0; i < _units; i++) {
      goal(activity(i) * typename Scalar::Value(_activity_regularization));
    }
  }

  return applyActivation(activity, _activation);
}

template <class Scalar>
void DenseLayer<Scalar>::serialize(
    const std::function<void(NeuralBase *, void *, size_t)> &fnc) {
  fnc(this, _bias.data(), _bias.bytes());
  fnc(this, _weights.data(), _weights.bytes());
}

// template class DenseLayer<double>;
// template class DenseLayer<float>;
//
// template class DenseLayer<Batch<double, 4>>;
// template class DenseLayer<Batch<double, 8>>;
// template class DenseLayer<Batch<double, 16>>;
//
// template class DenseLayer<Batch<float, 4>>;
// template class DenseLayer<Batch<float, 8>>;
// template class DenseLayer<Batch<float, 16>>;

// template class DenseLayer<Var<double>>;
// template class DenseLayer<Var<float>>;

template class DenseLayer<Var<Batch<double, 4>>>;
template class DenseLayer<Var<Batch<double, 8>>>;
template class DenseLayer<Var<Batch<double, 16>>>;

// template class DenseLayer<Var<Batch<float, 4>>>;
// template class DenseLayer<Var<Batch<float, 8>>>;
// template class DenseLayer<Var<Batch<float, 16>>>;

} // namespace tractor
