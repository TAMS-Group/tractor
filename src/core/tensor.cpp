// 2020-2024 Philipp Ruppel

#include <tractor/core/tensor.h>

#include <tractor/core/factory.h>
#include <tractor/core/lambda.h>
#include <tractor/core/log.h>
#include <tractor/core/profiler.h>
#include <tractor/core/var.h>

#include <unordered_map>

#include <boost/functional/hash.hpp>

namespace tractor {

static constexpr size_t g_tensor_alignment = 32;

size_t TensorShape::hash() const noexcept {
  size_t hash = 0;
  boost::hash_combine(hash, _data.size());
  for (auto &v : _data) {
    boost::hash_combine(hash, v);
  }
  return hash;
}

const std::string makeTensorName(const TypeInfo &element,
                                 const TensorShape &shape) {
  std::string name = element.name();
  for (auto &s : shape) {
    name += "_" + std::to_string(s);
  }
  return name;
}

TypeInfo makeTensorType(const TypeInfo &element_type,
                        const TensorShape &shape) {
  return TypeInfo::make(
      makeTensorName(element_type, shape),
      element_type.size() * shape.elementCount(),
      std::max(size_t(1), (element_type.alignment() + g_tensor_alignment - 1) /
                              g_tensor_alignment) *
          g_tensor_alignment);
}

const Operator *createTensorOpVariant(const Operator *element_op,
                                      const OpGroup &group,
                                      const TensorShape &shape) {
  if (!element_op) {
    TRACTOR_DEBUG("op not found");
    return nullptr;
  }

  std::vector<Operator::Argument> args;
  {
    for (auto &arg : element_op->arguments()) {
      const TypeInfo &t = makeTensorType(arg.typeInfo(), shape);
      if (arg.isInput())
        args.push_back(Operator::Argument::makeInput(t));
      else
        args.push_back(Operator::Argument::makeOutput(t));
    }
  }

  std::string tensor_op_name = element_op->name();
  for (auto &s : shape) {
    tensor_op_name += "_" + std::to_string(s);
  }

  size_t element_count = shape.elementCount();

  if (element_op->opType() == OpType(typeid(op_move *)) &&
      element_op->opMode() == OpType(typeid(compute *)) && args.size() == 2 &&
      args[0].typeInfo() == args[1].typeInfo()) {
    std::shared_ptr<ProfilerTrack> profiler_track =
        Profiler::instance()->track(std::make_shared<ProfilerTrack>(
            __PRETTY_FUNCTION__, "fastcopy " + tensor_op_name));
    size_t byte_count = args[0].typeInfo().size();
    return makePointerOp(
        tensor_op_name, element_op->label(), element_op->opMode(),
        element_op->opType(), group, args,
        [byte_count, profiler_track](const void *src, void *dest) {
          ProfilerScope profiler_scope(*profiler_track);
          std::memcpy(dest, src, byte_count);
        });
  }

  {
    std::shared_ptr<ProfilerTrack> profiler_track = Profiler::instance()->track(
        std::make_shared<ProfilerTrack>(__PRETTY_FUNCTION__, tensor_op_name));
    return makeListOperator(
        tensor_op_name, element_op->label(), element_op->opMode(),
        element_op->opType(), group, args,
        [element_op, element_count, profiler_track](void *base,
                                                    const uintptr_t *offsets) {
          ProfilerScope profiler_scope(*profiler_track);
          element_op->functionPointers().iterate(base, offsets, element_count);
        });
  }
}

const Operator *makeTensorOp(const Operator *element_op,
                             const TensorShape &tensor_shape) {
  static Factory::Key<const Operator *, TensorShape>::Value<const Operator *>
      factory([](const Operator *element_op, const TensorShape &shape) {
        std::string group_name = element_op->name();
        for (auto &s : shape) {
          group_name += "_" + std::to_string(s);
        }
        OpGroup group = makeOpGroup(group_name);

        auto *tensor_op = createTensorOpVariant(element_op, group, shape);
        if (!tensor_op) {
          throw std::runtime_error("failed to create tensor op");
        }
        if (!createTensorOpVariant(element_op->variant<forward>(), group,
                                   shape)) {
          throw std::runtime_error("failed to create forward tensor op");
        }
        if (!createTensorOpVariant(element_op->variant<reverse>(), group,
                                   shape)) {
          throw std::runtime_error("failed to create reverse tensor op");
        }
        if (auto *variant = element_op->tryFindVariant<prepare>()) {
          createTensorOpVariant(variant, group, shape);
        }

        return tensor_op;
      });
  return factory.get(element_op, tensor_shape);
}

void emitTensorOpImpl(
    const Operator *element_op,
    const std::initializer_list<const TensorInfo *> &tensor_infos,
    const std::initializer_list<void *> &tensor_data) {
  if (tensor_infos.size() == 0) {
    return;
  }
  if (tensor_infos.size() != tensor_data.size() ||
      element_op->argumentCount() != tensor_infos.size()) {
    throw std::invalid_argument("wrong number of arguments");
  }
  size_t argument_count = tensor_infos.size();
  const auto &shape = (*tensor_infos.begin())->shape();
  for (auto &inf : tensor_infos) {
    if (inf->shape() != shape) {
      throw std::invalid_argument(element_op->name() +
                                  " tensor shapes do not match");
    }
  }

  auto *tensor_op = makeTensorOp(element_op, shape);
  tensor_op->callIndirect((void **)std::data(tensor_data));
  if (auto *rec = Recorder::instance()) {
    rec->op(tensor_op);
    auto it_info = tensor_infos.begin();
    auto it_data = tensor_data.begin();
    for (size_t i = 0; i < argument_count; i++) {
      rec->arg((*it_info)->type(), *it_data);
      it_info++;
      it_data++;
    }
  }
}

std::ostream &operator<<(std::ostream &s, const TensorShape &v) {
  s << "[";
  for (size_t i = 0; i < v.dimensions(); i++) {
    if (i > 0) {
      s << ",";
    }
    s << v[i];
  }
  s << "]";
  return s;
}

TensorOperators::TensorOperators(const TypeInfo &element_type,
                                 const TypeInfo &tensor_type,
                                 const TensorShape &tensor_shape) {
  TRACTOR_DEBUG("create tensor operators " << tensor_type.name());
  _move = makeTensorOp(Operator::find<compute, op_move>({element_type}),
                       tensor_shape);
  _zero = makeTensorOp(Operator::find<compute, op_zero>({element_type}),
                       tensor_shape);
  _add = makeTensorOp(
      Operator::find<compute, op_add>({element_type, element_type}),
      tensor_shape);
}

TensorInfo::TensorInfo(const std::string &name, const TypeInfo &element_type,
                       const TensorShape &shape)
    : _name(name) {
  TRACTOR_DEBUG("create tensor info " << name);
  _type = TypeInfo::make(_name, element_type.size() * shape.elementCount(),
                         element_type.alignment());
  _shape = shape;
  _operators = TensorOperators(element_type, _type, shape);
}

const TensorInfo *TensorInfo::_make(const TypeInfo &element,
                                    const TensorShape &shape) {
  std::string name = makeTensorName(element, shape);
  static std::unordered_map<std::string, const TensorInfo *> registry;
  if (!registry[name]) {
    registry[name] = new TensorInfo(name, element, shape);
  }
  return registry[name];
}

}  // namespace tractor
