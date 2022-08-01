// (c) 2020-2022 Philipp Ruppel

#include <tractor/tensor/newtensor.h>

#include <unordered_map>

namespace tractor {

// void makeOp(const std::string &name, const std::string &label,
//             const OpMode &mode, const OpType &op, const OpGroup &group,
//             const std::vector<size_t> context,
//             const std::vector<Operator::Argument> &args,
//             void (*callback)(void *base, const uintptr_t *offsets)) {
//   struct Impl : Operator {
//     Impl(const std::string &name, const std::string &label, const OpMode
//     &mode,
//          const OpType &op, const OpGroup &group,
//          const std::vector<size_t> context,
//          const std::vector<Operator::Argument> &args,
//          void (*callback)(void *base, const uintptr_t *offsets))
//         : Operator(name, label, mode, op, group) {
//       std::cout << "make op " << name << std::endl;
//       _arguments = args;
//       _argument_count = args.size();
//       _functions.indirect = callback;
//       _functions.context = context;
//     }
//   };
//   static std::unordered_map<std::string, Impl *> map;
//   if (!map[name]) {
//     map[name] = new Impl(name, label, mode, op, group, context, args,
//     callback);
//   }
// }

// static std::vector<Argument> arguments() { return
// {Argument::make<Args>()...}; }

// template <class Functor> std::vector<Operator::Argument> makeArgumentList()
// {}

// template <class... Args> struct FunctorOperator : Operator {
//   std::function<Args...> functor;
//   template <size_t... Indices>
//   void init(const std::integer_sequence<size_t, Indices...> &indices) {
//     _functions.indirect = [](void *base, const uintptr_t *offsets) {
//       FunctorOperator *_this = (FunctorOperator *)offsets[0];
//       _this->functor(
//           *(typename std::decay<Args>::type *)(void *)((uint8_t *)base +
//                                                        offsets[Indices])...);
//     };
//   }
//   FunctorOperator(const std::string &name, const std::string &label,
//                   const OpMode &mode, const OpType &op, const OpGroup &group,
//                   const std::function<Args...> &functor)
//       : Operator(name, label, mode, op, group), functor(functor) {
//     std::cout << "make op " << name << std::endl;
//     _arguments =
//         std::vector<Operator::Argument>({Operator::Argument::make<Args>()...});
//     _argument_count = _arguments.size();
//     init(sizeof...(Args));
//     std::vector<uintptr_t> context{this};
//     _functions.context = context;
//   }
// };
// template <class... Args>
// void makeOp(const std::string &name, const std::string &label,
//             const OpMode &mode, const OpType &op, const OpGroup &group,
//             const std::function<Args...> &functor) {
//   static std::unordered_map<std::string, const Operator *> map;
//   if (!map[name]) {
//     map[name] =
//         new FunctorOperator<Args...>(name, label, mode, op, group, functor);
//   }
// }

TypeInfo makeTensorType(const TypeInfo &element, const TensorShape &shape) {

  std::string name = std::string() + "tensor_" + element.name();
  for (auto &s : shape) {
    name += "_" + std::to_string(s);
  }

  auto tensor_type = TypeInfo::make(name, element.size() * shape.elementCount(),
                                    element.alignment());

  // std::cout << OpMode(typeid(compute *)).name() << " "
  //           << OpGroup(typeid(op_move *)).name() << " "
  //           << Operator::Argument::makeInput(tensor_type).typeInfo().name()
  //           << " "
  //           << Operator::Argument::makeOutput(tensor_type).typeInfo().name()
  //           << std::endl;

  std::vector<uintptr_t> context;
  // context.push_back(element.size() * shape.elementCount());
  /*
  context.push_back(shape.dimensions());
  for (auto &s : shape) {
    context.push_back(s);
  }
  */

  std::cout << "make tensor move " << name << std::endl;

  std::string move_op_name = std::string() + "move_" + name;

  auto group = makeOpGroup(move_op_name);

  size_t size = element.size() * shape.elementCount();

  makePointerOp(
      move_op_name, "move", OpMode(typeid(compute *)),
      OpType(typeid(op_move *)), group,
      {
          Operator::Argument::makeInput(tensor_type),
          Operator::Argument::makeOutput(tensor_type),
      },
      std::function<void(const void *, void *)>([size](const void *a, void *b) {
        std::cout << "tensor move " << size << std::endl;
        std::memcpy(b, a, size);
      }));

  // makeOp(move_op_name, "move", OpMode(typeid(compute *)),
  //        OpType(typeid(op_move *)), makeOpGroup(move_op_name),
  //        [](void *base, const uintptr_t *offsets) {
  //          std::cout << "tensor move" << std::endl;
  //
  //        });

  // std::string move_op_name = "move_" + name;
  // makeOp(move_op_name, "move", OpMode(typeid(compute *)),
  //        OpType(typeid(op_move *)), makeOpGroup(move_op_name), context,
  //        {
  //            Operator::Argument::makeInput(tensor_type),
  //            Operator::Argument::makeOutput(tensor_type),
  //        },
  //        [](void *base, const uintptr_t *offsets) {
  //          std::cout << "tensor move" << std::endl;
  //          // std::memcpy(base + offsets[2], base + offsets[1], offsets[0]);
  //        });

  // auto *move_op = makeListOperator(
  //     "move_" + name, "move",
  //     {
  //         Operator::Argument::makeInput(tensor_type),
  //         Operator::Argument::makeOutput(tensor_type),
  //     },
  //     [](const void *base, const uintptr_t *offsets) {
  //       //
  //     },
  //     [](const void *base, const uintptr_t *offsets) {
  //       //
  //     },
  //     [](const void *base, const uintptr_t *offsets) {
  //       //
  //     });

  return tensor_type;
}

} // namespace tractor
