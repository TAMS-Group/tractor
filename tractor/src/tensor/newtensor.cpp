// (c) 2020-2022 Philipp Ruppel

#include <tractor/tensor/newtensor.h>

#include <unordered_map>

namespace tractor {

TensorOperators::TensorOperators(const TypeInfo &element_type,
                                 const TypeInfo &tensor_type,
                                 const TensorShape &tensor_shape,
                                 void (*add)(size_t, const void *, const void *,
                                             void *)) {

  size_t element_count = tensor_shape.elementCount();
  size_t byte_count = element_type.size() * tensor_shape.elementCount();

  _add = makePointerOp(
      std::string() + "add_" + tensor_type.name(), "add",
      OpType(typeid(op_add *)),
      {
          Operator::Argument::makeInput(tensor_type),
          Operator::Argument::makeInput(tensor_type),
          Operator::Argument::makeOutput(tensor_type),
      },
      [element_count, add](const void *a, const void *b, void *x) {
        // std::cout << " > tensor add " << element_count << std::endl;
        add(element_count, a, b, x);
      },
      [element_count, add](const void *a, const void *b, const void *x,
                           const void *da, const void *db, void *dx) {
        // std::cout << " > f tensor add " << element_count << std::endl;
        add(element_count, da, db, dx);
      },
      [byte_count](const void *a, const void *b, const void *x, void *da,
                   void *db, const void *dx) {
        // std::cout << " > r tensor add " << byte_count << std::endl;
        std::memcpy(da, dx, byte_count);
        std::memcpy(db, dx, byte_count);
      });

  _move = makePointerOp(
      std::string() + "move_" + tensor_type.name(), "move",
      OpType(typeid(op_move *)),
      {
          Operator::Argument::makeInput(tensor_type),
          Operator::Argument::makeOutput(tensor_type),
      },
      [byte_count](const void *a, void *x) {
        // std::cout << " > tensor move " << byte_count << std::endl;
        std::memcpy(x, a, byte_count);
      },
      [byte_count](const void *a, const void *x, const void *da, void *dx) {
        // std::cout << " > f tensor move " << byte_count << std::endl;
        std::memcpy(dx, da, byte_count);
      },
      [byte_count](const void *a, const void *x, void *da, const void *dx) {
        // std::cout << " > r tensor move " << byte_count << std::endl;
        std::memcpy(da, dx, byte_count);
      });

  _zero = makePointerOp(
      std::string() + "zero_" + tensor_type.name(), "zero",
      OpType(typeid(op_zero *)),
      {
          Operator::Argument::makeOutput(tensor_type),
      },
      [byte_count](void *x) {
        // std::cout << " > tensor zero " << byte_count << std::endl;
        std::memset(x, 0, byte_count);
      },
      [byte_count](const void *x, void *dx) {
        // std::cout << " > f tensor zero " << byte_count << std::endl;
        std::memset(dx, 0, byte_count);
      },
      [byte_count](const void *x, void *dx) {
        // std::cout << " > r tensor zero " << byte_count << std::endl;
      });
}

TensorInfo::TensorInfo(const std::string &name, const TypeInfo &element_type,
                       const TensorShape &shape,
                       void (*add)(size_t, const void *, const void *, void *))
    : _name(name) {
  _type = TypeInfo::make(_name, element_type.size() * shape.elementCount(),
                         element_type.alignment());
  _shape = shape;
  _operators = TensorOperators(element_type, _type, shape, add);
}

const TensorInfo *
TensorInfo::_make(const TypeInfo &element, const TensorShape &shape,
                  void (*add)(size_t, const void *, const void *, void *)) {
  std::string name = std::string() + "tensor_" + element.name();
  for (auto &s : shape) {
    name += "_" + std::to_string(s);
  }
  static std::unordered_map<std::string, const TensorInfo *> registry;
  if (!registry[name]) {
    registry[name] = new TensorInfo(name, element, shape, add);
  }
  return registry[name];
}

} // namespace tractor
