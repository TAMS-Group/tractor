// 2020-2024 Philipp Ruppel

#include <tractor/core/type.h>

#include <tractor/core/log.h>

#include <iostream>
#include <map>
#include <unordered_map>

namespace tractor {

TypeInfo TypeInfo::make(const std::string &name, size_t size,
                        size_t alignment) {
  struct DataEx : Data {
    std::string str;
  };
  static std::unordered_map<std::string, DataEx *> map;
  if (!map[name]) {
    DataEx *d = new DataEx();
    d->str = name;
    d->name = d->str.c_str();
    d->size = size;
    d->alignment = alignment;
    d->id = d;
    map[name] = d;
  }
  TypeInfo ret;
  ret._data = map[name];
  return ret;
}

const void *TypeInfo::Data::makeId(const std::type_info &type) {
  static std::unordered_map<std::type_index, char> id_map;
  return &id_map[type];
}

static std::map<TypeInfo, TypeInfo> &gradientTypeMap() {
  static std::map<TypeInfo, TypeInfo> m;
  return m;
}

const TypeInfo &TypeInfo::gradientType(const TypeInfo &type) {
  auto &m = gradientTypeMap();
  auto it = m.find(type);
  if (it != m.end()) {
    return it->second;
  } else {
    return type;
  }
}

void TypeInfo::registerGradientType(const TypeInfo &type,
                                    const TypeInfo &gradient) {
  TRACTOR_DEBUG("gradient type " << type.name() << " " << gradient.name());
  auto &m = gradientTypeMap();
  m[type] = gradient;
}

}  // namespace tractor
