#include <iostream>
#include <cassert>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.h"

using tensorflow::Tensor;
using tensorflow::TensorShape;

// 一些Tensor使用方式展示
// 资源来自于tensorflow的代码库
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_testutil.h
// 

// 单个变量组成的scalar类型
template <typename T>
tensorflow::Tensor AsScalar(const T& val) {
    tensorflow::Tensor ret(tensorflow::DataTypeToEnum<T>::value, {});
    ret.scalar<T>()() = val;
    return ret;
}

// Constructs a flat tensor with 'vals'.
template <typename T>
tensorflow::Tensor AsTensor(const std::vector<T>& vals) {
    tensorflow::Tensor ret(tensorflow::DataTypeToEnum<T>::value, {static_cast<tensorflow::int64>(vals.size())});
    std::copy_n(vals.data(), vals.size(), ret.flat<T>().data());
    return ret;
}

// 输入数据源为一维数组类型，然后可以根据shape进行变形
// 输入数据源的大小要和shape总数据匹配
template <typename T>
tensorflow::Tensor AsTensor(const std::vector<T>& vals, const tensorflow::TensorShape& shape) {
    tensorflow::Tensor ret;
    ret.CopyFrom(AsTensor(vals), shape);
    return ret;
}

static void new_line() {
    std::cout << "\n =============================== \n";
}

int main(int argc, char* argv[]) {

    {
        Tensor t1 = AsScalar(100);
        ::printf("t1.dims(): %d\n", t1.dims());
        // t1.dims(): 0
        ::printf("t1.scalar(): %d\n", t1.scalar<int>()());
        //t1.scalar(): 100
    }

    new_line();

    {
        std::vector<int> vec { 3, 8, 2, 9, 4};
        Tensor t1 = AsTensor(vec);
        ::printf("t1.dims(): %d, t1.dim_size(0): %d\n",
            t1.dims(), t1.dim_size(0));
        // t1.dims(): 1, t1.dim_size(0): 5

        auto t1_ft = t1.flat<int>();
        for(size_t i=0; i<t1.dim_size(0); ++i) {
            std::cout << "\t" << t1_ft(i);
        }
        std::cout << std::endl;
        // 	3	8	2	9	4

        auto t1_vec = t1.vec<int>();
        for(size_t i=0; i<t1.dim_size(0); ++i) {
            std::cout << "\t" << t1_vec(i);
        }
        std::cout << std::endl;
    }

    new_line();

    {
        std::vector<int> vec { 1, 2, 3, 4, 5, 6};
        Tensor t1 = AsTensor(vec, {3, 2}); // 3行2列
        ::printf("t1.dims(): %d, t1.dim_size(0): %d, t1.dim_size(1): %d\n",
            t1.dims(), t1.dim_size(0), t1.dim_size(1));
        // t1.dims(): 2, t1.dim_size(0): 3, t1.dim_size(1): 2

        // 一个简便的元素个数的统计返回
        ::printf("t1.NumElements(): %lld\n", t1.NumElements());
        // t1.NumElements(): 6

        auto t1_ft = t1.flat<int>();
        for(size_t i=0; i<t1.dim_size(0); ++i) {
            for(size_t j=0; j<t1.dim_size(1); ++j) {
                std::cout << "\t" << t1_ft(i * t1.dim_size(1) + j);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        //	1	2
        //  3	4
        //  5	6

        auto t1_mat = t1.matrix<int>();
        for(size_t i=0; i<t1.dim_size(0); ++i) {
            for(size_t j=0; j<t1.dim_size(1); ++j) {
                std::cout << "\t" << t1_mat(i, j);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    new_line();

    {
        std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8};
        Tensor t1 = AsTensor(vec, {4, 2});
        auto t1_mat = t1.matrix<int>();
        for(size_t i=0; i<t1.dim_size(0); ++i) {
            for(size_t j=0; j<t1.dim_size(1); ++j) {
                std::cout << "\t" << t1_mat(i, j);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        //  1	2
        //	3	4
        //	5	6
        //	7	8

        // 无论是Slice还是SubSlice，都是引用复用底层的存储结构
        // Slice得到的结果是const的，不能修改；序该元素只能从最原始的Tensor操作

        // 左闭右开的结构 
        // [dim_start, dim_limit)
        Tensor t1_slice = t1.Slice(1, 3);
        std::cout << "t1_slice: " << t1_slice.DebugString() << std::endl;
        // t1_slice: Tensor<type: int32 shape: [2,2] values: [3 4][5...]...>

        // 只按第一个维度切分
        Tensor t1_subslice = t1.SubSlice(1);
        std::cout << "t1_subslice: " << t1_subslice.DebugString() << std::endl;
        // t1_subslice: Tensor<type: int32 shape: [2] values: 3 4>

        std::cout << "t1.IsAligned(): " << t1.IsAligned() << std::endl; // true
        std::cout << "t1_slice.IsAligned(): " << t1_slice.IsAligned() << std::endl; // false
        std::cout << "t1_subslice.IsAligned(): " << t1_subslice.IsAligned() << std::endl; // false

        // 性能优化，减少分配？？？

        // 原始Tensor的数据变了，这边引用的Slice也看见了这个变化
        auto ft = t1.flat<int>();
        ft(2) = 100;
        std::cout << "t1_slice: " << t1_slice.DebugString() << std::endl;
        // t1_slice: Tensor<type: int32 shape: [2,2] values: [100 4][5...]...>

    }

    std::cout << "[INFO] Tensor finished." << std::endl;
    return EXIT_SUCCESS;
}
