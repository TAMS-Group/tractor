// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <unordered_set>
#include <vector>

#include <tractor/core/eigen.h>
#include <tractor/core/engine.h>
#include <tractor/core/error.h>
#include <tractor/core/profiler.h>

#include <Eigen/Sparse>

#include <omp.h>

namespace tractor {

class Program;

class SparsitySet {
  std::unordered_set<size_t> _indices;

public:
  void insert(size_t row) { _indices.insert(row); }
  auto &data() const { return _indices; }
  SparsitySet &operator|=(const SparsitySet &other) {
    for (auto &i : other._indices) {
      _indices.insert(i);
    }
    return *this;
  }
};

inline bool operator&(const SparsitySet &a, const SparsitySet &b) {
  for (auto &i : b.data()) {
    if (a.data().find(i) != a.data().end()) {
      return true;
    }
  }
  return false;
}

class SparsityMatrix {
  size_t _rows = 0;
  size_t _cols = 0;
  std::vector<SparsitySet> _columns;

public:
  SparsityMatrix(size_t rows, size_t cols) { init(rows, cols); }
  SparsityMatrix(const Program &program, size_t stride);
  void init(size_t rows, size_t cols) {
    _rows = rows;
    _cols = cols;
    _columns.resize(cols);
  }
  auto &col(size_t i) { return _columns.at(i); }
  auto &col(size_t i) const { return _columns.at(i); }
  size_t rows() const { return _rows; }
  size_t cols() const { return _cols; }
  void insert(size_t row, size_t col) {
    TRACTOR_ASSERT(row < _rows);
    TRACTOR_ASSERT(col < _cols);
    return _columns.at(col).insert(row);
  }
  template <class T>
  Eigen::SparseMatrix<T> toEigenSparseMatrix(const T &nonzero = T(1)) const {
    std::vector<Eigen::Triplet<T>> triplets;
    for (size_t col = 0; col < _cols; col++) {
      for (size_t row : _columns[col].data()) {
        triplets.emplace_back(row, col, nonzero);
      }
    }
    Eigen::SparseMatrix<T> ret(_rows, _cols);
    ret.setFromTriplets(triplets.begin(), triplets.end());
    ret.makeCompressed();
    return ret;
  }
};

class SparsityBase {

protected:
  SparsityMatrix _sparsity_matrix;

  class OutputGroup {
    size_t _input_index;
    std::vector<size_t> _output_indices;

  public:
    OutputGroup(size_t input_index, const SparsitySet &col)
        : _input_index(input_index) {
      for (auto &row : col.data()) {
        _output_indices.push_back(row);
      }
    }
    size_t inputIndex() const { return _input_index; }
    auto &outputIndices() const { return _output_indices; }
  };

  class InputGroup {
    SparsitySet _active_set;
    std::vector<size_t> _input_indices;
    std::vector<OutputGroup> _output_groups;

  public:
    bool tryAdd(size_t input_index, const SparsitySet &col) {
      if (_active_set & col) {
        return false;
      }
      _active_set |= col;
      _input_indices.push_back(input_index);
      _output_groups.emplace_back(input_index, col);
      return true;
    }
    void add(size_t input_index, const SparsitySet &col) {
      TRACTOR_ASSERT(tryAdd(input_index, col));
    }
    auto &inputIndices() const { return _input_indices; }
    auto &outputGroups() const { return _output_groups; }
  };

  std::vector<InputGroup> _input_groups;

public:
  SparsityBase(const Program &program, size_t stride);
  auto &sparsityMatrix() const { return _sparsity_matrix; }
};

template <class T> class SparseMatrixBuilder : public SparsityBase {
  std::shared_ptr<const Engine> _engine;
  std::shared_ptr<const Executable> _executable;
  struct ThreadData {
    Eigen::Matrix<T, Eigen::Dynamic, 1> input_vector, output_vector;
    std::shared_ptr<Memory> memory;
    Buffer input_buffer, output_buffer;
  };
  std::vector<ThreadData> _thread_data;

public:
  SparseMatrixBuilder(const std::shared_ptr<const Engine> &engine,
                      const Program &program,
                      const std::shared_ptr<const Executable> &executable)
      : SparsityBase(program, sizeof(T)), _engine(engine),
        _executable(executable) {
    for (size_t i = 0; i < omp_get_max_threads(); i++) {
      ThreadData tda;
      tda.memory = _engine->createMemory();
      _thread_data.push_back(tda);
    }
  }

  size_t complexity() const { return _input_groups.size(); }

  Eigen::SparseMatrix<T> build(const std::shared_ptr<Memory> &memory) {

    std::vector<Eigen::Triplet<T>> triplets;

    size_t input_group_count = _input_groups.size();

#pragma omp parallel
    {
      // TRACTOR_DEBUG(_memories.size() << " " << omp_get_thread_num());
      auto &tda = _thread_data.at(omp_get_thread_num());
      {
        TRACTOR_PROFILER("spmb copy memory");
        memory->copyTo(tda.memory);
      }
    }

#pragma omp parallel for
    for (size_t input_group_index = 0; input_group_index < input_group_count;
         input_group_index++) {

      auto &input_group = _input_groups.at(input_group_index);

      auto &tda = _thread_data.at(omp_get_thread_num());

      {
        TRACTOR_PROFILER("spmb input vector");
        tda.input_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(
            _executable->inputBufferSize() / sizeof(T));
        for (size_t i : input_group.inputIndices()) {
          tda.input_vector(i) = T(1);
        }
      }

      {
        TRACTOR_PROFILER("spmb set input vector");
        //_executable->inputVector(tda.input_vector, tda.memory);
        tda.input_buffer.fromVector(tda.input_vector);
        _executable->input(tda.input_buffer, tda.memory);
      }

      {
        TRACTOR_PROFILER("spmb run matrix program");
        _executable->execute(tda.memory);
      }

      {
        TRACTOR_PROFILER("spmb get output vector");
        //_executable->outputVector(tda.memory, tda.output_vector);
        _executable->output(tda.memory, tda.output_buffer);
        tda.output_buffer.toVector(tda.output_vector);
      }

#pragma omp critical
      {
        TRACTOR_PROFILER("spmb collect coefficients");
        for (auto &output_group : input_group.outputGroups()) {
          size_t col = output_group.inputIndex();
          for (size_t row : output_group.outputIndices()) {
            triplets.emplace_back(row, col, tda.output_vector(row));
          }
        }
      }
    }

    Eigen::SparseMatrix<T> ret(_sparsity_matrix.rows(),
                               _sparsity_matrix.cols());
    {
      TRACTOR_PROFILER("spmb assemble matrix");
      ret.setFromTriplets(triplets.begin(), triplets.end());
      ret.makeCompressed();
    }
    return ret;
  }

  // Eigen::SparseMatrix<T> build(const std::shared_ptr<Executable> &executable,
  //                              const std::shared_ptr<Memory> &memory) {
  //
  //   std::vector<Eigen::Triplet<T>> triplets;
  //
  //   for (auto &input_group : _input_groups) {
  //
  //     Eigen::Matrix<T, Eigen::Dynamic, 1> input_vector =
  //         Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(
  //             executable->inputBufferSize() / sizeof(T));
  //
  //     for (size_t i : input_group.inputIndices()) {
  //       input_vector(i) = T(1);
  //     }
  //
  //     executable->inputVector(input_vector, memory);
  //
  //     executable->execute(memory);
  //
  //     Eigen::Matrix<T, Eigen::Dynamic, 1> output_vector;
  //     executable->outputVector(memory, output_vector);
  //
  //     for (auto &output_group : input_group.outputGroups()) {
  //       size_t col = output_group.inputIndex();
  //       for (size_t row : output_group.outputIndices()) {
  //         triplets.emplace_back(row, col, output_vector(row));
  //       }
  //     }
  //   }
  //
  //   Eigen::SparseMatrix<T> ret(_sparsity_matrix.rows(),
  //                              _sparsity_matrix.cols());
  //   ret.setFromTriplets(triplets.begin(), triplets.end());
  //   ret.makeCompressed();
  //   return ret;
  // }
};

} // namespace tractor
