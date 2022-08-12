// // (c) 2020-2022 Philipp Ruppel
//
// #include "operator.h"
//
// #include <pybind11/pybind11.h>
//
// namespace tractor {
//
// template <class Impl, class Mode, class Op, class Group, class Scalar>
// class OperatorImpl : public Operator {
//   template <class... Args> struct Init {
//     template <class Ret, size_t... Indices> struct Looper {
//       // static void loop(void *base, const uintptr_t *offsets,
//       //                  size_t iterations) TRACTOR_FAST {
//       //   for (size_t i = 0; i < iterations; i++) {
//       //     *(Ret *)(void *)((uint8_t *)base + offsets[sizeof...(Indices)])
//       =
//       //         Impl::call(
//       //             *(typename std::decay<Args>::type
//       //                   *)(void *)((uint8_t *)base +
//       offsets[Indices])...);
//       //     offsets += sizeof...(Indices) + 1;
//       //   }
//       // }
//       static inline void
//       iterateImpl(size_t iterations, Ret *ret,
//                   typename std::decay<Args>::type *...args) TRACTOR_FAST {
//         for (size_t i = 0; i < iterations; i++) {
//           ret[i] = Impl::call(args[i]...);
//         }
//       }
//       static void iterate(void *base, const uintptr_t *offsets,
//                           size_t iterations) TRACTOR_FAST {
//         iterateImpl(
//             iterations,
//             (Ret *)(void *)((uint8_t *)base + offsets[sizeof...(Indices)]),
//             ((typename std::decay<Args>::type *)(void *)((uint8_t *)base +
//                                                          offsets[Indices]))...);
//       }
//       static void indirect(void *base, const uintptr_t *offsets) TRACTOR_FAST
//       {
//         *(Ret *)(void *)((uint8_t *)base + offsets[sizeof...(Indices)]) =
//             Impl::call(*(typename std::decay<Args>::type
//                              *)(void *)((uint8_t *)base +
//                              offsets[Indices])...);
//       }
//       // static void direct(typename std::decay<Args>::type *...args,
//       //                    Ret *ret) TRACTOR_FAST {
//       //   *ret = Impl::call(*args...);
//       // }
//       static std::vector<Argument> arguments() TRACTOR_SLOW {
//         return {Argument::make<Args>()..., Argument::make<Ret &>()};
//       }
//     };
//     template <size_t... Indices> struct Looper<void, Indices...> {
//       // static void loop(void *base, const uintptr_t *offsets,
//       //                  size_t iterations) {
//       //   for (size_t i = 0; i < iterations; i++)
//       //     TRACTOR_FAST {
//       //       Impl::call(*(typename std::decay<Args>::type
//       //                        *)(void *)((uint8_t *)base +
//       //                        offsets[Indices])...);
//       //       offsets += sizeof...(Indices);
//       //     }
//       // }
//       static inline void
//       iterateImpl(size_t iterations,
//                   typename std::decay<Args>::type *...args) TRACTOR_FAST {
//         for (size_t i = 0; i < iterations; i++) {
//           Impl::call(args[i]...);
//         }
//       }
//       static void iterate(void *base, const uintptr_t *offsets,
//                           size_t iterations) TRACTOR_FAST {
//         iterateImpl(
//             iterations,
//             ((typename std::decay<Args>::type *)(void *)((uint8_t *)base +
//                                                          offsets[Indices]))...);
//       }
//       static void indirect(void *base, const uintptr_t *offsets) TRACTOR_FAST
//       {
//         Impl::call(
//             *(typename std::decay<Args>::type *)(void *)((uint8_t *)base +
//                                                          offsets[Indices])...);
//       }
//       // static void
//       // direct(typename std::decay<Args>::type *...args) TRACTOR_FAST {
//       //   Impl::call(*args...);
//       // }
//       static std::vector<Argument> arguments() TRACTOR_SLOW {
//         return {Argument::make<Args>()...};
//       }
//     };
//   };
//   typedef typename ReturnType<decltype(&Impl::call)>::Type Return;
//   template <size_t... Indices, class... Args>
//   void init(const std::integer_sequence<size_t, Indices...> &indices,
//             std::tuple<Args...> *) {
//     typedef Init<Args...> _Init;
//     typedef typename _Init::template Looper<Return, Indices...> _Loop;
//     // _functions.loop = &_Loop::loop;
//     _functions.iterate = &_Loop::iterate;
//     _functions.indirect = &_Loop::indirect;
//     // _functions.direct = reinterpret_cast<const void *>(&_Loop::direct);
//     _arguments = _Loop::arguments();
//   }
//   typedef typename RawArgumentTuple<decltype(&Impl::call)>::Type
//   ArgumentTuple;
//
//   template <class Ret, class... Args> struct Pythonizer {
//     static void pythonize(const Operator *op, pybind11::module &m,
//                           Ret (*func)(Args &...)) {
//       m.def(op->label().c_str(), [op](typename MakeVar<Args>::Type &...args)
//       {
//         Var<Ret> ret;
//         op->invoke(&args..., &ret);
//         recordOperation(op, &args..., &ret);
//         return ret;
//       });
//     }
//   };
//   template <class... Args> struct Pythonizer<void, Args...> {
//     static void pythonize(const Operator *op, pybind11::module &m,
//                           void (*func)(Args &...)) {
//       m.def(op->label().c_str(), [op](typename MakeVar<Args>::Type &...args)
//       {
//         op->invoke(&args...);
//         recordOperation(op, &args...);
//       });
//     }
//   };
//   template <class Ret, class... Args>
//   static void pythonizeImpl(const Operator *op, pybind11::module &m,
//                             Ret (*func)(Args &...)) {
//     Pythonizer<Ret, Args...>::pythonize(op, m, func);
//     m.def(op->label().c_str(), [op](typename MakeTensor<Args>::Type &...args)
//     {
//       return TensorOpCaller<Ret>::call(op, args...);
//     });
//   }
//
//   template <class X, class T> struct PythonizerFilter {
//     static void pythonize(const Operator *op, pybind11::module &m) {}
//   };
//   template <class X> struct PythonizerFilter<X, compute> {
//     static void pythonize(const Operator *op, pybind11::module &m) {
//       pythonizeImpl(op, m, &Impl::call);
//     }
//   };
//   virtual void pythonize(pybind11::module &m) const override {
//     PythonizerFilter<int, Mode>::pythonize(this, m);
//   }
//
// public:
//   OperatorImpl(const std::string &name, const std::string &label)
//       : Operator(name, label, OpMode(typeid(Mode *)), OpType(typeid(Op *)),
//                  OpGroup(typeid(Group *))) {
//     constexpr size_t argument_count = std::tuple_size<ArgumentTuple>::value;
//     init(std::make_index_sequence<argument_count>(), (ArgumentTuple
//     *)nullptr);
//   }
//   static const Operator *instance(const char *name, const char *label) {
//     static const Operator *instance = [name, label]() {
//       auto *instance =
//           tryFind(OpMode(typeid(Mode *)), OpGroup(typeid(Group *)));
//       if (!instance) {
//         instance = new OperatorImpl(name, label);
//       }
//       return instance;
//     }();
//     return instance;
//   }
// };
//
// #undef TRACTOR_OP_TYPED
//
// #define TRACTOR_OP_TYPED(mode, prefix, name, args, impl, scalar, postfix) \
//                                                                                \
//   class op_##name; \
//   struct op_##prefix##name##_##postfix##_impl_1 { \
//     typedef scalar T; \
//     typedef BatchScalar<scalar>::Type S; \
//     static inline auto call args TRACTOR_FAST impl; \
//   }; \
//                                                                                \
//   struct scalar##postfix##_group; \
//                                                                                \
//   const Operator *op_##prefix##name##_##postfix##_inst = OperatorImpl< \
//       op_##prefix##name##_##postfix##_impl_1, mode, op_##name, \
//       std::tuple<op_##name *, scalar##postfix##_group *>, \
//       scalar>::instance(TRACTOR_STRINGIFY(prefix##name##_##postfix), \
//                         TRACTOR_STRINGIFY(name)); \
//                                                                                \
//   struct op_##prefix##name##_##postfix##_impl_2 \
//       : op_##prefix##name##_##postfix##_impl_1 { \
//     static decltype(op_##prefix##name##_##postfix##_inst) instance(); \
//   }; \
//   decltype(op_##prefix##name##_##postfix##_inst) \
//       op_##prefix##name##_##postfix##_impl_2::instance() { \
//     return op_##prefix##name##_##postfix##_inst; \
//   } \
//                                                                                \
//   namespace op_##prefix##name##_##postfix##_ns { \
//     typedef scalar T; \
//     typedef BatchScalar<scalar>::Type S; \
//     static op_##prefix##name##_##postfix##_impl_2 \
//         *op_##prefix##name##_overload args; \
//   } \ using op_##prefix##name##_##postfix##_ns::op_##prefix##name##_overload;
//
// } // namespace tractor
