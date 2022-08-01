// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <cstring>
#include <string>
#include <type_traits>
#include <typeindex>

namespace tractor {

template <class T> struct TypeInfoID {
  static void id() {}
};

class TypeInfo {
  struct Data {
    const void *id = nullptr;
    size_t size = 0;
    size_t alignment = 0;
    const char *name = nullptr;
    static const void *makeId(const std::type_info &type);
    template <class T> static Data make() {
      Data data;
      data.id = makeId(typeid(T));
      data.size = sizeof(T);
      data.alignment = std::alignment_of<T>::value;
      data.name = typeid(T).name();
      return data;
    }
  };
  const Data *_data = nullptr;

public:
  inline TypeInfo() {}
  template <class T> static inline TypeInfo get() {
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

// class TypeInfoData {
//   size_t _size = 0;
//   std::type_index _type = typeid(void);
//   size_t _alignment = 0;
//
// public:
//   template <class T> static const TypeInfoData *get() {
//     static TypeInfoData ret;
//     return &ret;
//   }
// };

// class TypeInfo {
//   size_t _size = 0;
//   std::type_index _type = typeid(void);
//   size_t _alignment = 0;
//   // const TypeInfoData *_data = nullptr;
//
// public:
//   inline TypeInfo() {}
//   // inline TypeInfo(const size_t &size, const std::type_index &type,
//   //                 const size_t &alignment)
//   //     : _size(size), _type(type), _alignment(alignment) {}
//   template <class T> static inline TypeInfo get() {
//     typedef typename std::decay<T>::type X;
//     TypeInfo t;
//     t._size = sizeof(X);
//     t._type = typeid(X);
//     t._alignment = std::alignment_of<X>::value;
//     return t;
//   }
//   inline size_t size() const { return _size; }
//   // inline const std::type_index &type() const { return _type; }
//   inline const char *name() const { return _type.name(); }
//   static const TypeInfo &gradientType(const TypeInfo &type);
//   const TypeInfo &gradientType() const { return gradientType(*this); }
//   static void registerGradientType(const TypeInfo &type,
//                                    const TypeInfo &gradient);
//   size_t alignment() const { return _alignment; }
//   inline bool operator==(const TypeInfo &b) const { return _type == b._type;
//   } inline bool operator!=(const TypeInfo &b) const { return _type !=
//   b._type; } inline bool operator<(const TypeInfo &b) const { return _type <
//   b._type; } inline bool operator>(const TypeInfo &b) const { return _type >
//   b._type; } inline bool operator<=(const TypeInfo &b) const { return _type
//   <= b._type; } inline bool operator>=(const TypeInfo &b) const { return
//   _type >= b._type; }
// };

#define TRACTOR_GRADIENT_TYPE_CONCAT_2(a, b) a##b

#define TRACTOR_GRADIENT_TYPE_CONCAT(a, b) TRACTOR_GRADIENT_TYPE_CONCAT_2(a, b)

#define TRACTOR_GRADIENT_TYPE(type, gradient)                                  \
  static int TRACTOR_GRADIENT_TYPE_CONCAT(g_gradient_type_reg_, __COUNTER__) = \
      (TypeInfo::registerGradientType(TypeInfo::get<type>(),                   \
                                      TypeInfo::get<gradient>()),              \
       0);

#define TRACTOR_GRADIENT_TYPE_SPECIALIZE(type, gradient, scalar)               \
  namespace TRACTOR_GRADIENT_TYPE_CONCAT(g_gradient_type_reg_, __COUNTER__) {  \
    typedef scalar T;                                                          \
    TRACTOR_GRADIENT_TYPE(type, gradient)                                      \
  }

#define TRACTOR_GRADIENT_TYPE_TEMPLATE(type, gradient)                         \
  TRACTOR_GRADIENT_TYPE_SPECIALIZE(type, gradient, double)                     \
  TRACTOR_GRADIENT_TYPE_SPECIALIZE(type, gradient, float)                      \
  TRACTOR_GRADIENT_TYPE_SPECIALIZE(type, gradient, int)

} // namespace tractor
