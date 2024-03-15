// 2020-2024 Philipp Ruppel

#pragma once

#include <cstring>
#include <string>
#include <type_traits>
#include <typeindex>

namespace tractor {

template <class T>
struct TypeInfoID {
  static void id() {}
};

template <class T>
struct TypeNameHelper {
  static const char *name() { return typeid(T).name(); }
};

#define TRACTOR_TYPE_NAME_OVERRIDE(type, namestr) \
  template <>                                     \
  struct TypeNameHelper<type> {                   \
    static const char *name() { return namestr; } \
  };

TRACTOR_TYPE_NAME_OVERRIDE(float, "float")
TRACTOR_TYPE_NAME_OVERRIDE(double, "double")
TRACTOR_TYPE_NAME_OVERRIDE(uint64_t, "uint64")
TRACTOR_TYPE_NAME_OVERRIDE(uint32_t, "uint32")
TRACTOR_TYPE_NAME_OVERRIDE(int64_t, "int64")
TRACTOR_TYPE_NAME_OVERRIDE(int32_t, "int32")

class TypeInfo {
  struct Data {
    const void *id = nullptr;
    size_t size = 0;
    size_t alignment = 0;
    const char *name = nullptr;
    static const void *makeId(const std::type_info &type);
    template <class T>
    static Data make() {
      Data data;
      data.id = makeId(typeid(T));
      data.size = sizeof(T);
      data.alignment = std::alignment_of<T>::value;
      // data.name = typeid(T).name();
      data.name = TypeNameHelper<T>::name();
      return data;
    }
  };
  const Data *_data = nullptr;

 public:
  inline TypeInfo() {}
  template <class T>
  static inline TypeInfo get() {
    static Data data = Data::make<typename std::decay<T>::type>();
    TypeInfo t;
    t._data = &data;
    return t;
  }
  static TypeInfo make(const std::string &name, size_t size, size_t alignment);
  inline size_t size() const { return _data->size; }
  inline const char *name() const { return _data->name; }
  static const TypeInfo &gradientType(const TypeInfo &type);
  const TypeInfo &gradientType() const { return gradientType(*this); }
  static void registerGradientType(const TypeInfo &type,
                                   const TypeInfo &gradient);
  size_t alignment() const { return _data->alignment; }
  inline bool operator==(const TypeInfo &b) const {
    return _data->id == b._data->id;
  }
  inline bool operator!=(const TypeInfo &b) const {
    return _data->id != b._data->id;
  }
  inline bool operator<(const TypeInfo &b) const {
    return _data->id < b._data->id;
  }
  inline bool operator>(const TypeInfo &b) const {
    return _data->id > b._data->id;
  }
  inline bool operator<=(const TypeInfo &b) const {
    return _data->id <= b._data->id;
  }
  inline bool operator>=(const TypeInfo &b) const {
    return _data->id >= b._data->id;
  }
};

#define TRACTOR_GRADIENT_TYPE_CONCAT_2(a, b) a##b

#define TRACTOR_GRADIENT_TYPE_CONCAT(a, b) TRACTOR_GRADIENT_TYPE_CONCAT_2(a, b)

#define TRACTOR_GRADIENT_TYPE(type, gradient)                                  \
  static int TRACTOR_GRADIENT_TYPE_CONCAT(g_gradient_type_reg_, __COUNTER__) = \
      (TypeInfo::registerGradientType(TypeInfo::get<type>(),                   \
                                      TypeInfo::get<gradient>()),              \
       0);

#define TRACTOR_GRADIENT_TYPE_SPECIALIZE(type, gradient, scalar)              \
  namespace TRACTOR_GRADIENT_TYPE_CONCAT(g_gradient_type_reg_, __COUNTER__) { \
  typedef scalar T;                                                           \
  TRACTOR_GRADIENT_TYPE(type, gradient)                                       \
  }

#define TRACTOR_GRADIENT_TYPE_TEMPLATE(type, gradient)     \
  TRACTOR_GRADIENT_TYPE_SPECIALIZE(type, gradient, double) \
  TRACTOR_GRADIENT_TYPE_SPECIALIZE(type, gradient, float)  \
  TRACTOR_GRADIENT_TYPE_SPECIALIZE(type, gradient, int)

}  // namespace tractor
