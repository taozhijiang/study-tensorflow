#ifndef __START_TF_SIMPLE_TENSOR_UTIL__
#define __START_TF_SIMPLE_TENSOR_UTIL__

#include <vector>

#include <tensorflow/core/framework/tensor.h>

template <typename T>
tensorflow::Tensor AsTensor(const std::vector<T> &vals) {
  tensorflow::Tensor ret(tensorflow::DataTypeToEnum<T>::value,
                         {static_cast<int>(vals.size())});
  std::copy_n(vals.data(), vals.size(), ret.flat<T>().data());
  return ret;
}

template <typename T>
tensorflow::Tensor AsTensor(const std::vector<T> &vals,
                            const tensorflow::TensorShape &shape) {
  tensorflow::Tensor ret;
  ret.CopyFrom(AsTensor(vals), shape);
  return ret;
}

#endif  // __START_TF_SIMPLE_TENSOR_UTIL__