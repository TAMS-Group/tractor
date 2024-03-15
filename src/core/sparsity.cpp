// 2020-2024 Philipp Ruppel

#include <tractor/core/sparsity.h>

#include <tractor/core/operator.h>
#include <tractor/core/program.h>

namespace tractor {

template <class PortArray>
static size_t findPortSize(const PortArray &ports) {
  size_t size = 0;
  for (auto &port : ports) {
    size = std::max(size, port.offset() + port.size());
  }
  return size;
}

template <class Mask>
class SparsityVector {
  std::vector<Mask> _data;
  size_t _stride = 0;

 public:
  SparsityVector() {}
  SparsityVector(size_t bytecount, size_t stride) { clear(bytecount, stride); }
  size_t stride() const { return _stride; }
  void clear(size_t bytecount, size_t stride) {
    _data.clear();
    _data.resize((bytecount + stride - 1) / stride, Mask(0));
    _stride = stride;
  }
  Mask read(size_t address, size_t size) const {
    size_t begin = address / _stride;
    size_t end = (address + size + _stride - 1) / _stride;
    Mask ret = Mask(0);
    for (size_t i = begin; i < end; i++) {
      ret |= _data.at(i);
    }
    return ret;
  }
  void write(size_t address, size_t size, const Mask &mask) {
    size_t begin = address / _stride;
    size_t end = (address + size + _stride - 1) / _stride;
    for (size_t i = begin; i < end; i++) {
      _data.at(i) |= mask;
    }
  }
  auto &element(size_t i) { return _data.at(i); }
  auto &element(size_t i) const { return _data.at(i); }
};

template <class Mask>
static SparsityVector<Mask> propagateSparsity(
    const Program &program, const SparsityVector<Mask> &input_sparsity) {
  SparsityVector<Mask> memory_sparsity(program.memorySize(),
                                       input_sparsity.stride());
  for (auto &input : program.inputs()) {
    Mask sparsity = input_sparsity.read(input.offset(), input.size());
    memory_sparsity.write(input.address(), input.size(), sparsity);
  }

  for (auto &instruction : program.instructions()) {
    auto *op = instruction.op();
    Mask sparsity = Mask(0);
    for (size_t i = 0; i < op->argumentCount(); i++) {
      if (op->arg(i).isInput()) {
        sparsity |= memory_sparsity.read(instruction.arg(i), op->arg(i).size());
      }
    }
    for (size_t i = 0; i < op->argumentCount(); i++) {
      if (op->arg(i).isOutput()) {
        memory_sparsity.write(instruction.arg(i), op->arg(i).size(), sparsity);
      }
    }
  }

  SparsityVector<Mask> output_sparsity(findPortSize(program.outputs()),
                                       input_sparsity.stride());
  for (auto &output : program.outputs()) {
    Mask sparsity = memory_sparsity.read(output.address(), output.size());
    output_sparsity.write(output.offset(), output.size(), sparsity);
  }
  return output_sparsity;
}

template <class Mask>
inline Mask makeMask(size_t i) {
  TRACTOR_ASSERT(i < (sizeof(Mask) * 8));
  return (Mask(1) << i);
}

template <class Mask>
static void buildSparsityMatrix(const Program &program, size_t stride,
                                SparsityMatrix &sparsity_matrix) {
  TRACTOR_DEBUG("finding sparsity pattern");

  size_t input_bytes = findPortSize(program.inputs());
  size_t output_bytes = findPortSize(program.outputs());

  sparsity_matrix.init((output_bytes + stride - 1) / stride,
                       (input_bytes + stride - 1) / stride);

  const size_t col_batch_size = sizeof(Mask) * 8;
  const size_t col_batches =
      (sparsity_matrix.cols() + col_batch_size - 1) / col_batch_size;

#pragma omp parallel for
  for (size_t col_batch = 0; col_batch < col_batches; col_batch++) {
    size_t col_begin = col_batch * col_batch_size;
    size_t col_end =
        std::min(col_begin + col_batch_size, sparsity_matrix.cols());

    SparsityVector<Mask> input_sparsity(input_bytes, stride);
    for (size_t col_index = col_begin; col_index < col_end; col_index++) {
      input_sparsity.element(col_index) = makeMask<Mask>(col_index - col_begin);
    }

    SparsityVector output_sparsity =
        propagateSparsity<Mask>(program, input_sparsity);

#pragma omp critical
    {
      for (size_t col_index = col_begin; col_index < col_end; col_index++) {
        for (size_t row_index = 0; row_index < sparsity_matrix.rows();
             row_index++) {
          if (output_sparsity.element(row_index) &
              makeMask<Mask>(col_index - col_begin)) {
            sparsity_matrix.insert(row_index, col_index);
          }
        }
      }
    }
  }
}

SparsityMatrix::SparsityMatrix(const Program &program, size_t stride) {
  buildSparsityMatrix<uint64_t>(program, stride, *this);
}

SparsityBase::SparsityBase(const Program &program, size_t stride)
    : _sparsity_matrix(program, stride) {
  TRACTOR_DEBUG("analyzing sparsity pattern");

  std::unordered_set<size_t> col_set;
  for (size_t col = 0; col < _sparsity_matrix.cols(); col++) {
    col_set.insert(col);
  }

  while (!col_set.empty()) {
    InputGroup input_group;

    {
      size_t first_col = *col_set.begin();
      col_set.erase(first_col);
      input_group.add(first_col, _sparsity_matrix.col(first_col));
    }

    {
      auto col_it = col_set.begin();
      while (col_it != col_set.end()) {
        size_t col_index = *col_it;
        bool added =
            input_group.tryAdd(col_index, _sparsity_matrix.col(col_index));
        if (added) {
          col_set.erase(col_it++);
        } else {
          col_it++;
        }
      }
    }

    _input_groups.push_back(input_group);
  }

  TRACTOR_DEBUG("sparsity pattern analyzed");
}

}  // namespace tractor
