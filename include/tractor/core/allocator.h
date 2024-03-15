// 2020-2024 Philipp Ruppel

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace tractor {

void *tractor_malloc(size_t size);
void tractor_free(void *ptr);

class TypeInfo;
class Program;

class Allocator {
  size_t _top = 32;

 public:
  Allocator() {}
  Allocator(size_t top) : _top(top) {}
  void clear() { _top = 32; }
  size_t top() const { return _top; }
  size_t alloc(const TypeInfo &type);
  void keep(const Program &program);
  void init(size_t i) { _top = i; }
  void apply(Program &program);
};

template <class T>
struct AlignedStdAlloc : std::allocator<T> {
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;
  typedef T value_type;
  template <class U>
  struct rebind {
    typedef AlignedStdAlloc<U> other;
  };
  AlignedStdAlloc() : std::allocator<T>() {}
  AlignedStdAlloc(const AlignedStdAlloc &other) : std::allocator<T>(other) {}
  template <class U>
  AlignedStdAlloc(const AlignedStdAlloc<U> &other) : std::allocator<T>(other) {}
  ~AlignedStdAlloc() {}
  pointer allocate(size_type num, const void *hint = 0) const {
    return static_cast<pointer>(tractor_malloc(num * sizeof(T)));
  }
  void deallocate(pointer p, size_type) const { tractor_free(p); }
};

template <class T>
using AlignedStdVector = std::vector<T, AlignedStdAlloc<T>>;

}  // namespace tractor
