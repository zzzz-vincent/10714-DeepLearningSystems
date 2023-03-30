

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

/**
 * Operation used for elementwise and scalar operations
 */
scalar_t ADD(scalar_t a, scalar_t b) {return a + b;}
scalar_t DIV(scalar_t a, scalar_t b) {return a / b;}
scalar_t MUL(scalar_t a, scalar_t b) {return a * b;}
scalar_t EQ(scalar_t a, scalar_t b) {return a == b;}
scalar_t GE(scalar_t a, scalar_t b) {return a >= b;}
scalar_t POWER(scalar_t a, scalar_t b) {return std::pow(a, b);}
scalar_t LOG(scalar_t a) {return std::log(a);}
scalar_t EXP(scalar_t a) {return std::exp(a);}
scalar_t TANH(scalar_t a) {return std::tanh(a);}
scalar_t MAX(scalar_t a, scalar_t b) {return std::max(a, b);}


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};

void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

size_t GetIndex(size_t idx, const std::vector<int32_t>& shape, const std::vector<int32_t>& strides) {
  /**
   * Given index of compact array, find its out_index
   * in a non-compact array with `shape` and `strides` 
   */
  size_t out_index = 0;
  size_t cur_ptr = shape.size() - 1;
  while (idx != 0) {
    size_t val = idx % shape[cur_ptr];
    idx = idx / shape[cur_ptr];
    out_index += strides[cur_ptr] * val;
    cur_ptr--;
  }
  return out_index;
}


void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   * 
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  for (size_t idx = 0; idx < out->size; idx++) {
    size_t a_idx = GetIndex(idx, shape, strides);
    out->ptr[idx] = a.ptr[offset + a_idx];
  }
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  for (size_t idx = 0; idx < a.size; idx++) {
    size_t a_idx = GetIndex(idx, shape, strides);
    out->ptr[offset + a_idx] = a.ptr[idx];
  }
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  for (size_t idx = 0; idx < size; idx++) {
    size_t a_idx = GetIndex(idx, shape, strides);
    out->ptr[offset + a_idx] = val;
  }
}

void EwiseOp(const AlignedArray& a, AlignedArray* out, std::function<scalar_t(scalar_t)> operation) {
  /**
   * Perform element-wise operation correspondings entires in a.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = operation(a.ptr[i]);
  }
}

void EwiseOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, std::function<scalar_t(scalar_t, scalar_t)> operation) {
  /**
   * Perform element-wise operation correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = operation(a.ptr[i], b.ptr[i]);
  }
}

void ScalarOp(const AlignedArray& a, scalar_t val, AlignedArray* out, std::function<scalar_t(scalar_t, scalar_t)> operation) {
  /**
   * Perform element-wise operation correspondings entires in a and a scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = operation(a.ptr[i], val);
  }
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  EwiseOp(a, b, out, &ADD);
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  ScalarOp(a, val, out, &ADD);
}

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the multiply of correspondings entires in a and b.
   */
  EwiseOp(a, b, out, &MUL);
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the multiply of corresponding entry in a plus the scalar val.
   */
  ScalarOp(a, val, out, &MUL);
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the div of correspondings entires in a and b.
   */
  EwiseOp(a, b, out, &DIV);
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the div of corresponding entry in a plus the scalar val.
   */
  ScalarOp(a, val, out, &DIV);
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the power of corresponding entry in a plus the scalar val.
   */
  ScalarOp(a, val, out, &POWER);
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the max of correspondings entires in a and b.
   */
  EwiseOp(a, b, out, &MAX);
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the max of corresponding entry in a plus the scalar val.
   */
  ScalarOp(a, val, out, &MAX);
}

void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the equal of correspondings entires in a and b.
   */
  EwiseOp(a, b, out, &EQ);
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the equal of corresponding entry in a plus the scalar val.
   */
  ScalarOp(a, val, out, &EQ);
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the greater_or_qeual of correspondings entires in a and b.
   */
  EwiseOp(a, b, out, &GE);
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the greater_or_qeual of corresponding entry in a plus the scalar val.
   */
  ScalarOp(a, val, out, &GE);
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  /**
   * Set entries in out to be the log of correspondings entires in a.
   */
  EwiseOp(a, out, &LOG);
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  /**
   * Set entries in out to be the log of correspondings entires in a.
   */
  EwiseOp(a, out, &EXP);
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  /**
   * Set entries in out to be the log of correspondings entires in a.
   */
  EwiseOp(a, out, &TANH);
}

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: coolumns of b / out
   */

   out->size = m * p;
   Fill(out, 0);
   for (size_t m_i = 0; m_i < m; m_i++) {
     for (size_t p_i = 0; p_i < p; p_i++) {
       for (size_t n_i = 0; n_i < n; n_i++) {
         out->ptr[m_i * p + p_i] += a.ptr[m_i * n + n_i] * b.ptr[n_i * p + p_i];
       }
     }
   }
}

inline void AlignedDot(const float* __restrict__ a, 
                       const float* __restrict__ b, 
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement 
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and 
   * out don't have any overlapping memory (which is necessary in order for vector operations to be 
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b, 
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the 
   * compiler that the input array siwll be aligned to the appropriate blocks in memory, which also 
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  for (size_t i = 0; i < TILE; i++) {
    for (size_t j = 0; j < TILE; j++) {
      for (size_t k = 0; k < TILE; k++) {
        out[i*TILE+j] += a[i*TILE+k] * b[k*TILE+j];
      }
    }
  }
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   * 
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   * 
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: coolumns of b / out
   * 
   */
  out->size = m * p;
  Fill(out, 0);
  for (size_t m_i = 0; m_i < m; m_i += TILE) {
     for (size_t p_i = 0; p_i < p; p_i += TILE) {
       for (size_t n_i = 0; n_i < n; n_i += TILE) {
         AlignedDot(a.ptr    + m_i * n + n_i * TILE,
                    b.ptr    + n_i * p + p_i * TILE,
                    out->ptr + m_i * p + p_i * TILE);
       }
     }
   }
}

void ReduceOp(const AlignedArray& a, AlignedArray* out, size_t step, std::function<scalar_t(scalar_t, scalar_t)> operation) {
  for (size_t i = 0; i < out->size; i++) {
    for (size_t j = 0; j < step; j++) {
      size_t idx = step * i + j;
      if (j == 0) {
        out->ptr[i] = a.ptr[idx];
      } else {
        out->ptr[i] = operation(out->ptr[i], a.ptr[idx]);
      }
    }
  }
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */

  ReduceOp(a, out, reduce_size, &MAX);
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */

  ReduceOp(a, out, reduce_size, &ADD);
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);
  
  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}