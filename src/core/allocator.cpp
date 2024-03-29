// 2020-2024 Philipp Ruppel

#include <tractor/core/allocator.h>
#include <tractor/core/program.h>
#include <tractor/core/type.h>

#include <cstdlib>

namespace tractor {

void *tractor_malloc(size_t size) {
  auto *ret = ::aligned_alloc(32, size);

  if (!ret) {
    throw std::bad_alloc();
  }

  if ((((uintptr_t)ret) % 32) != 0) {
    throw std::runtime_error("aligned alloc failed " +
                             std::to_string((uintptr_t)ret));
  }

  return ret;
}

void tractor_free(void *ptr) {
  if (!ptr) {
    throw std::runtime_error("free null");
  }
  free(ptr);
}

size_t Allocator::alloc(const TypeInfo &type) {
  while (_top % type.alignment()) {
    _top++;
  }
  size_t ret = _top;
  _top += type.size();
  return ret;
}

void Allocator::keep(const Program &program) { _top = program.memorySize(); }

void Allocator::apply(Program &program) { program.updateMemorySize(_top); }

}  // namespace tractor
