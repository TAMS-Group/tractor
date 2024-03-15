// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/batch.h>
#include <tractor/core/platform.h>
#include <tractor/core/recorder.h>
#include <tractor/core/tensor.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <tuple>

namespace pybind11 {
class module_;
using module = module_;
}  // namespace pybind11

#ifdef TRACTOR_IMPLEMENT_OPS
#include <pybind11/pybind11.h>
#endif

namespace tractor {

class VarBase {};

template <class T>
class alignas(T) Var : public VarBase {
  T _x = T();

 public:
  typedef T Value;
  inline Var() {
    if (auto *inst = Recorder::instance()) {
      inst->constant(this);
    }
  }
  inline Var(const T &v) : _x(v) {
    if (auto *inst = Recorder::instance()) {
      inst->constant(this);
    }
  }
  inline Var(const Var &other) {
    _x = other._x;
    if (auto *inst = Recorder::instance()) {
      inst->move(&other._x, &_x);
    }
  }
  template <class X, class Y = decltype(T(std::declval<X>()))>
  explicit inline Var(const X &v) : _x(T(v)) {
    if (auto *inst = Recorder::instance()) {
      inst->constant(this);
    }
  }
  // explicit operator T() const { return _x; }
  inline T &value() { return _x; }
  inline const T &value() const { return _x; }
  inline Var(Var &&other) {
    _x = other._x;
    if (auto *inst = Recorder::instance()) {
      inst->rewrite(&other._x, &_x);
    }
  }
  inline Var &operator=(const Var &other) {
    _x = other._x;
    if (auto *inst = Recorder::instance()) {
      inst->move(&other._x, &_x);
    }
    return *this;
  }
  inline Var &operator=(Var &&other) {
    _x = other._x;
    if (auto *inst = Recorder::instance()) {
      inst->rewrite(&other._x, &_x);
    }
    return *this;
  }
};

template <class T>
struct MakeVar {
  typedef const Var<T> Type;
};
template <class T>
struct MakeVar<const T> {
  typedef const Var<T> Type;
};
template <class T>
struct MakeVar<const T &> {
  typedef const Var<T> Type;
};
template <class T>
struct MakeVar<T &> {
  typedef Var<T> Type;
};
template <class T>
struct MakeVar<T *> {
  typedef const Var<T *> Type;
};

class OpTypeBase {
 protected:
  std::type_index _type_index = typeid(void);
  const void *_pointer = nullptr;
  std::tuple<std::type_index, const void *> pack() const {
    return std::make_tuple(_type_index, _pointer);
  }

 public:
  inline OpTypeBase(const std::type_index &type_index)
      : _type_index(type_index) {}
  inline OpTypeBase(const void *pointer) : _pointer(pointer) {}
  inline bool operator<(const OpTypeBase &b) const { return pack() < b.pack(); }
  inline bool operator==(const OpTypeBase &b) const {
    return pack() == b.pack();
  }
  inline bool operator!=(const OpTypeBase &b) const {
    return pack() != b.pack();
  }
  size_t hash() const {
    return std::hash<std::type_index>()(_type_index) ^
           std::hash<const void *>()(_pointer);
  }
  const char *name() const { return _type_index.name(); }
};

class OpType : public OpTypeBase {
 public:
  OpType(const std::type_index &type_index) : OpTypeBase(type_index) {}
  OpType(const std::type_info &type_index) : OpTypeBase(type_index) {}
  inline OpType(const void *pointer) : OpTypeBase(pointer) {}
};

class OpGroup : public OpTypeBase {
 public:
  OpGroup(const std::type_index &type_index) : OpTypeBase(type_index) {}
  OpGroup(const std::type_info &type_index) : OpTypeBase(type_index) {}
  inline OpGroup(const void *pointer) : OpTypeBase(pointer) {}
};

class OpMode : public OpTypeBase {
 public:
  OpMode(const std::type_index &type_index) : OpTypeBase(type_index) {}
  OpMode(const std::type_info &type_index) : OpTypeBase(type_index) {}
};
}  // namespace tractor

template <>
struct std::hash<tractor::OpType> {
  std::size_t operator()(const tractor::OpType &v) const noexcept {
    return v.hash();
  }
};
template <>
struct std::hash<tractor::OpGroup> {
  std::size_t operator()(const tractor::OpGroup &v) const noexcept {
    return v.hash();
  }
};
template <>
struct std::hash<tractor::OpMode> {
  std::size_t operator()(const tractor::OpMode &v) const noexcept {
    return v.hash();
  }
};

namespace tractor {

typedef void (*LoopFunction)(void *base, const uintptr_t *offsets,
                             size_t iterations);

typedef void (*OpFunction)(void *base, const uintptr_t *offsets);

struct OperatorFunctions {
  std::vector<uint64_t> context;
  // LoopFunction loop = nullptr;
  LoopFunction iterate = nullptr;
  OpFunction indirect = nullptr;
  const void *direct = nullptr;
};

template <class Functor>
class RawArgumentTuple {
  template <class Ret, class... Args>
  static std::tuple<Args...> *getArgumentTuple(Ret (*)(Args...)) {
    return nullptr;
  }

 public:
  typedef typename std::decay<decltype(*getArgumentTuple(
      *(Functor *)nullptr))>::type Type;
};

template <class Functor>
class ArgumentValueTuple {
  template <class Ret, class... Args>
  static std::tuple<typename std::decay<Args>::type...> *getArgumentTuple(
      Ret (*)(Args...)) {
    return nullptr;
  }

 public:
  typedef typename std::decay<decltype(*getArgumentTuple(
      *(Functor *)nullptr))>::type Type;
};

template <class Functor>
class ReturnType {
  template <class Ret, class... Args>
  static Ret *getReturnType(Ret (*)(Args...)) {
    return nullptr;
  }

 public:
  typedef
      typename std::decay<typename std::remove_pointer<decltype(getReturnType(
          *(Functor *)nullptr))>::type>::type Type;
};

template <class T>
struct IsVar {
  static constexpr bool value = false;
};
template <class T>
struct IsVar<Var<T>> {
  static constexpr bool value = true;
};

template <class... TT>
struct AnyVar {};
template <class T, class... TT>
struct AnyVar<T, TT...> {
  static constexpr bool value =
      (IsVar<typename std::decay<T>::type>::value || AnyVar<TT...>::value);
};
template <>
struct AnyVar<> {
  static constexpr bool value = false;
};

class OperatorModeMap {
  std::vector<const Operator *> _ops;

 public:
  static size_t index(const OpMode &type);
  inline auto at(size_t i) const { return i < _ops.size() ? _ops[i] : nullptr; }
  void put(size_t i, const Operator *op) {
    _ops.resize(std::max(_ops.size(), i + 1), nullptr);
    _ops[i] = op;
  }
};

class Operator {
 public:
  class Argument {
    // size_t _size = 0;
    bool _is_const = false;
    TypeInfo _type;

   public:
    template <class T>
    static Argument make() {
      Argument ret;
      // ret._size = sizeof(typename std::decay<T>::type);
      ret._is_const =
          std::is_convertible<const typename std::decay<T>::type &, T>::value;
      ret._type = TypeInfo::get<T>();
      return ret;
    }
    static Argument makeInput(const TypeInfo &type) {
      Argument ret;
      ret._type = type;
      ret._is_const = true;
      return ret;
    }
    static Argument makeOutput(const TypeInfo &type) {
      Argument ret;
      ret._type = type;
      ret._is_const = false;
      return ret;
    }
    size_t size() const { return _type.size(); }
    bool isInput() const { return _is_const; }
    bool isOutput() const { return !_is_const; }
    // const std::type_index &type() const { return _type.type(); }
    const TypeInfo &typeInfo() const { return _type; }
    Argument makeReverse() const {
      Argument ret = *this;
      ret._is_const = !_is_const;
      return ret;
    }
    Argument makeInput() const {
      Argument ret = *this;
      ret._is_const = true;
      return ret;
    }
    Argument makeOutput() const {
      Argument ret = *this;
      ret._is_const = false;
      return ret;
    }
  };

 private:
  std::string _name, _label;
  OpMode _mode;
  OpType _op;
  const OperatorModeMap *_map = nullptr;

 protected:
  OperatorFunctions _functions;
  // size_t _argument_count = 0;
  std::vector<Argument> _arguments;
  static const Operator *tryFind(const OpMode &mode, const OpGroup &group);
  Operator(const std::string &name, const std::string &label,
           const OpMode &mode, const OpType &op, const OpGroup &group);
  virtual ~Operator();
  // template <class Mode, class Op, class... Args>
  // static const Operator *tryFind(const Args &...args) {
  //   return tryFind(OpMode(typeid(Mode *)), OpType(typeid(Op *)), {args...});
  // }

 public:
  Operator(const Operator &) = delete;
  Operator &operator=(const Operator &) = delete;
  template <class T>
  inline bool isMode() const {
    return _mode == OpMode(typeid(T *));
  }
  inline const std::string &name() const { return _name; }
  inline const std::string &label() const { return _label; }
  inline OperatorFunctions functionPointers() const { return _functions; }
  void callIndirect(void *base, uintptr_t *offsets) const;
  void callIndirect(void **args) const;
  void invoke(const void *first, ...) const;
  inline size_t argumentCount() const { return _arguments.size(); }
  inline size_t argumentSize(size_t i) const { return _arguments[i].size(); }
  inline auto arguments() const { return ArrayRef<const Argument>(_arguments); }
  inline const Argument &arg(size_t i) const { return _arguments.at(i); }
  template <class T>
  inline const Operator *tryFindVariant() const {
    size_t index = OperatorModeMap::index(OpMode(typeid(T *)));
    auto *op = _map->at(index);
    return op;
  }
  template <class T>
  inline const Operator *variant() const {
    auto *op = tryFindVariant<T>();
    if (!op) {
      throw std::runtime_error(std::string() + "variant not found: " +
                               typeid(T *).name() + " " + name());
    }
    return op;
  }
  static const Operator *tryFind(const std::string &name);
  static const Operator *find(const std::string &name);
  static const Operator *tryFind(const OpMode &mode, const OpType &op,
                                 const std::initializer_list<TypeInfo> &args);
  static const Operator *find(const OpMode &mode, const OpType &op,
                              const std::initializer_list<TypeInfo> &args);
  template <class Mode, class Op>
  static const Operator *tryFind(const std::initializer_list<TypeInfo> &args) {
    return tryFind(OpMode(typeid(Mode *)), OpType(typeid(Op *)), args);
  }
  template <class Mode, class Op>
  static const Operator *find(const std::initializer_list<TypeInfo> &args) {
    return find(OpMode(typeid(Mode *)), OpType(typeid(Op *)), args);
  }
  const OpType &opType() const { return _op; }
  const OpMode &opMode() const { return _mode; }
  template <class Op>
  bool is() const {
    return _op == OpType(typeid(Op *));
  }
  static std::vector<const Operator *> all();
  virtual void pythonize(pybind11::module &) const {}
};

template <class T>
struct ArgumentConverter {
  static inline const T &map(const Var<T> &v) { return v.value(); }
  static inline T &map(Var<T> &v) { return v.value(); }
};

template <class Ret, class Op>
struct Caller {
  template <class... Args>
  static inline Var<Ret> call2(Args &...args) {
    Var<Ret> ret;
    ret.value() = Op::call(args...);
    recordOperation(Op::instance(), &args..., &ret.value());
    return std::move(ret);
  }
  template <class... ImplArgs, class... Args>
  static inline Var<Ret> call(std::tuple<ImplArgs...> *, Args &...args) {
    return std::move(call2(ArgumentConverter<ImplArgs>::map(args)...));
  }
};
template <class Op>
struct Caller<void, Op> {
  template <class... Args>
  static inline void call2(Args &...args) {
    Op::call(args...);
    recordOperation(Op::instance(), &args...);
  }
  template <class... ImplArgs, class... Args>
  static inline void call(std::tuple<ImplArgs...> *, Args &...args) {
    call2(ArgumentConverter<ImplArgs>::map(args)...);
  }
};

#ifdef TRACTOR_IMPLEMENT_OPS

template <class Impl, class Mode, class Op, class Group, class Scalar>
class OperatorImpl : public Operator {
  template <class... Args>
  struct Init {
    template <class Ret, size_t... Indices>
    struct Looper {
      // static void loop(void *base, const uintptr_t *offsets,
      //                  size_t iterations) TRACTOR_FAST {
      //   for (size_t i = 0; i < iterations; i++) {
      //     *(Ret *)(void *)((uint8_t *)base + offsets[sizeof...(Indices)]) =
      //         Impl::call(
      //             *(typename std::decay<Args>::type
      //                   *)(void *)((uint8_t *)base + offsets[Indices])...);
      //     offsets += sizeof...(Indices) + 1;
      //   }
      // }
      static inline void iterateImpl(size_t iterations, Ret *ret,
                                     typename std::decay<Args>::type *...args)
          TRACTOR_FAST {
        for (size_t i = 0; i < iterations; i++) {
          ret[i] = Impl::call(args[i]...);
        }
      }
      static void iterate(void *base, const uintptr_t *offsets,
                          size_t iterations) TRACTOR_FAST {
        iterateImpl(
            iterations,
            (Ret *)(void *)((uint8_t *)base + offsets[sizeof...(Indices)]),
            ((typename std::decay<Args>::type *)(void *)((uint8_t *)base +
                                                         offsets[Indices]))...);
      }
      static void indirect(void *base, const uintptr_t *offsets) TRACTOR_FAST {
        *(Ret *)(void *)((uint8_t *)base + offsets[sizeof...(Indices)]) =
            Impl::call(*(typename std::decay<Args>::type
                             *)(void *)((uint8_t *)base + offsets[Indices])...);
      }
      static void direct(typename std::decay<Args>::type *...args,
                         Ret *ret) TRACTOR_FAST {
        *ret = Impl::call(*args...);
      }
      static std::vector<Argument> arguments() TRACTOR_SLOW {
        return {Argument::make<Args>()..., Argument::make<Ret &>()};
      }
    };
    template <size_t... Indices>
    struct Looper<void, Indices...> {
      // static void loop(void *base, const uintptr_t *offsets,
      //                  size_t iterations) {
      //   for (size_t i = 0; i < iterations; i++)
      //     TRACTOR_FAST {
      //       Impl::call(*(typename std::decay<Args>::type
      //                        *)(void *)((uint8_t *)base +
      //                        offsets[Indices])...);
      //       offsets += sizeof...(Indices);
      //     }
      // }
      static inline void iterateImpl(size_t iterations,
                                     typename std::decay<Args>::type *...args)
          TRACTOR_FAST {
        for (size_t i = 0; i < iterations; i++) {
          Impl::call(args[i]...);
        }
      }
      static void iterate(void *base, const uintptr_t *offsets,
                          size_t iterations) TRACTOR_FAST {
        iterateImpl(
            iterations,
            ((typename std::decay<Args>::type *)(void *)((uint8_t *)base +
                                                         offsets[Indices]))...);
      }
      static void indirect(void *base, const uintptr_t *offsets) TRACTOR_FAST {
        Impl::call(
            *(typename std::decay<Args>::type *)(void *)((uint8_t *)base +
                                                         offsets[Indices])...);
      }
      static void direct(typename std::decay<Args>::type *...args)
          TRACTOR_FAST {
        Impl::call(*args...);
      }
      static std::vector<Argument> arguments() TRACTOR_SLOW {
        return {Argument::make<Args>()...};
      }
    };
  };
  typedef typename ReturnType<decltype(&Impl::call)>::Type Return;
  template <size_t... Indices, class... Args>
  void init(const std::integer_sequence<size_t, Indices...> &indices,
            std::tuple<Args...> *) {
    typedef Init<Args...> _Init;
    typedef typename _Init::template Looper<Return, Indices...> _Loop;
    // _functions.loop = &_Loop::loop;
    _functions.iterate = &_Loop::iterate;
    _functions.indirect = &_Loop::indirect;
    _functions.direct = reinterpret_cast<const void *>(&_Loop::direct);
    _arguments = _Loop::arguments();
  }
  typedef typename RawArgumentTuple<decltype(&Impl::call)>::Type ArgumentTuple;

  template <class Ret, class... Args>
  struct Pythonizer {
    static void pythonize(const Operator *op, pybind11::module &m,
                          Ret (*func)(Args &...)) {
      m.def(op->label().c_str(), [op](typename MakeVar<Args>::Type &...args) {
        Var<Ret> ret;
        op->invoke(&args..., &ret);
        recordOperation(op, &args..., &ret);
        return ret;
      });
    }
  };
  template <class... Args>
  struct Pythonizer<void, Args...> {
    static void pythonize(const Operator *op, pybind11::module &m,
                          void (*func)(Args &...)) {
      m.def(op->label().c_str(), [op](typename MakeVar<Args>::Type &...args) {
        op->invoke(&args...);
        recordOperation(op, &args...);
      });
    }
  };
  template <class Ret, class... Args>
  static void pythonizeImpl(const Operator *op, pybind11::module &m,
                            Ret (*func)(Args &...)) {
    Pythonizer<Ret, Args...>::pythonize(op, m, func);
    m.def(op->label().c_str(), [op](typename MakeTensor<Args>::Type &...args) {
      return TensorOpCaller<Ret>::call(op, args...);
    });
  }

  template <class X, class T>
  struct PythonizerFilter {
    static void pythonize(const Operator *op, pybind11::module &m) {}
  };
  template <class X>
  struct PythonizerFilter<X, compute> {
    static void pythonize(const Operator *op, pybind11::module &m) {
      pythonizeImpl(op, m, &Impl::call);
    }
  };
  virtual void pythonize(pybind11::module &m) const override {
    PythonizerFilter<int, Mode>::pythonize(this, m);
  }

 public:
  OperatorImpl(const std::string &name, const std::string &label)
      : Operator(name, label, OpMode(typeid(Mode *)), OpType(typeid(Op *)),
                 OpGroup(typeid(Group *))) {
    constexpr size_t argument_count = std::tuple_size<ArgumentTuple>::value;
    init(std::make_index_sequence<argument_count>(), (ArgumentTuple *)nullptr);
  }
  static const Operator *instance(const char *name, const char *label) {
    static const Operator *instance = [name, label]() {
      auto *instance =
          tryFind(OpMode(typeid(Mode *)), OpGroup(typeid(Group *)));
      if (!instance) {
        instance = new OperatorImpl(name, label);
      }
      return instance;
    }();
    return instance;
  }
};

#endif

template <class T>
class Var;

template <class T>
struct OverloadSelector {
  template <class U,
            std::enable_if_t<std::is_convertible<T, U>::value, int> Z = 0>
  operator const U &() {
    return *(const U *)nullptr;
  }
};
template <class T>
struct OverloadSelector<const Var<T> &> {
  operator const T &() { return *(const T *)nullptr; }
};
template <class T>
struct OverloadSelector<Var<T> &> {
  operator T &() { return *(T *)nullptr; }
};
template <class T>
struct OverloadSelector<Var<T>> {
  operator const T &() { return *(const T *)nullptr; }
};

#define TRACTOR_STRINGIFY(x) #x

#ifdef TRACTOR_IMPLEMENT_OPS

#define TRACTOR_OP_TYPED(mode, prefix, name, args, impl, scalar, postfix)  \
                                                                           \
  class op_##name;                                                         \
  struct op_##prefix##name##_##postfix##_impl_1 {                          \
    typedef scalar T;                                                      \
    typedef BatchScalar<scalar>::Type S;                                   \
    static inline auto call args TRACTOR_FAST impl;                        \
  };                                                                       \
                                                                           \
  struct scalar##postfix##_group;                                          \
                                                                           \
  struct op_##prefix##name##_##postfix##_impl_2                            \
      : op_##prefix##name##_##postfix##_impl_1 {                           \
    static const Operator *instance();                                     \
  };                                                                       \
                                                                           \
  __attribute__((weak))                                                    \
  const Operator *op_##prefix##name##_##postfix##_impl_2_x = OperatorImpl< \
      op_##prefix##name##_##postfix##_impl_1, mode, op_##name,             \
      std::tuple<op_##name *, scalar##postfix##_group *>,                  \
      scalar>::instance(TRACTOR_STRINGIFY(prefix##name##_##postfix),       \
                        TRACTOR_STRINGIFY(name));                          \
                                                                           \
  __attribute__((weak))                                                    \
  const Operator *op_##prefix##name##_##postfix##_impl_2::instance() {     \
    return op_##prefix##name##_##postfix##_impl_2_x;                       \
  }                                                                        \
                                                                           \
  namespace op_##prefix##name##_##postfix##_ns {                           \
    typedef scalar T;                                                      \
    typedef BatchScalar<scalar>::Type S;                                   \
    static op_##prefix##name##_##postfix##_impl_2                          \
        *op_##prefix##name##_overload args;                                \
  }                                                                        \
  using op_##prefix##name##_##postfix##_ns::op_##prefix##name##_overload;

#else

#define TRACTOR_OP_TYPED(mode, prefix, name, args, impl, scalar, postfix) \
                                                                          \
  class op_##name;                                                        \
  struct op_##prefix##name##_##postfix##_impl_1 {                         \
    typedef scalar T;                                                     \
    typedef BatchScalar<scalar>::Type S;                                  \
    static inline auto call args TRACTOR_SLOW impl;                       \
  };                                                                      \
                                                                          \
  struct scalar##postfix##_group;                                         \
                                                                          \
  struct op_##prefix##name##_##postfix##_impl_2                           \
      : op_##prefix##name##_##postfix##_impl_1 {                          \
    static const Operator *instance();                                    \
  };                                                                      \
                                                                          \
  namespace op_##prefix##name##_##postfix##_ns {                          \
    typedef scalar T;                                                     \
    typedef BatchScalar<scalar>::Type S;                                  \
    static op_##prefix##name##_##postfix##_impl_2                         \
        *op_##prefix##name##_overload args;                               \
  }                                                                       \
  using op_##prefix##name##_##postfix##_ns::op_##prefix##name##_overload;

#endif

#define TRACTOR_OP_IMPL(mode, prefix, name, args, impl, postfix)         \
  TRACTOR_OP_TYPED(mode, prefix, name, args, impl, float, postfix##f)    \
  TRACTOR_OP_TYPED(mode, prefix, name, args, impl, double, postfix##d)   \
  TRACTOR_OP_TYPED(mode, prefix, name, args, impl, Batch4f, postfix##4f) \
  TRACTOR_OP_TYPED(mode, prefix, name, args, impl, Batch4d, postfix##4d)

// TRACTOR_OP_TYPED(mode, prefix, name, args, impl, uint64_t, postfix##i)       \
// TRACTOR_OP_TYPED(mode, prefix, name, args, impl, Batch8d, postfix##8d)
// TRACTOR_OP_TYPED(mode, prefix, name, args, impl, Batch16d, postfix##16d)

// template <class T> struct IsBatch { static constexpr bool value = false; };

class BatchBase {
 public:
  virtual ~BatchBase() {}
  virtual const void *vdata() const = 0;
  virtual void *vdata() = 0;
};

template <class T>
class TypedBatchBase : public BatchBase {
 public:
  virtual const T *data() const = 0;
  virtual T *data() = 0;
  virtual const void *vdata() const override { return data(); }
  virtual void *vdata() override { return data(); }
};

// static void isBatchTypeHelper(const std::initializer_list<BatchBase> &args)
// {}

#define TRACTOR_VAR_OP(name)                                                 \
  template <class... Args,                                                   \
            std::enable_if_t<AnyVar<Args...>::value, int> X = 0,             \
            class Impl = typename std::decay<decltype(*op_##name##_overload( \
                OverloadSelector<Args>()...))>::type,                        \
            class ImplArgs =                                                 \
                typename ArgumentValueTuple<decltype(&Impl::call)>::Type,    \
            class Ret = decltype(Impl::call(OverloadSelector<Args>()...)),   \
            decltype(Impl::call(OverloadSelector<Args>()...)) *Y = nullptr>  \
  inline auto name(Args &&...args) {                                         \
    return Caller<Ret, Impl>::call((ImplArgs *)nullptr, args...);            \
  }                                                                          \
                                                                             \
  template <class... Args,                                                   \
            class TensorCheck =                                              \
                decltype(checkAllTensorStatic(std::declval<Args>()...)),     \
            class Impl = typename std::decay<decltype(*op_##name##_overload( \
                *std::declval<Args>().data()...))>::type,                    \
            class Ret = typename std::decay<decltype(Impl::call(             \
                *std::declval<Args>().data()...))>::type>                    \
  inline auto name(Args &&...args) {                                         \
    return TensorOpCaller<Ret>::call(Impl::instance(), args...);             \
  }

#define TRACTOR_OP(name, args, impl)             \
  TRACTOR_OP_IMPL(compute, , name, args, impl, ) \
  TRACTOR_VAR_OP(name)

#define TRACTOR_D(mode, name, args, impl) \
  TRACTOR_OP_IMPL(mode, mode##_, name, args, impl, )

#define TRACTOR_OP_T(postfix, name, args, impl) \
  TRACTOR_OP_IMPL(compute, , name, args, impl, postfix##_)

#define TRACTOR_D_T(mode, postfix, name, args, impl) \
  TRACTOR_OP_IMPL(mode, mode##_, name, args, impl, postfix##_)

#define TRACTOR_D_LOOP(mode, name, args, args2, impl) \
  TRACTOR_D(mode, name, args, {                       \
    typedef S T;                                      \
    auto f = [] args {                                \
      typedef S T;                                    \
      impl                                            \
    };                                                \
    makeBatchLoop(f).run args2;                       \
  })

}  // namespace tractor
