cmake_minimum_required (VERSION 2.8.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wno-shadow -Wno-sign-compare")
set(CMAKE_CXX_STANDARD 11)

project(start_tf_mnist)

# 编译结果中的库和可执行文件的路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

set(TENSORFLOW_PREFIX
    "/Users/taozj/Dropbox/repos/machine_learning/TENSORFLOW/tensorflow-branch-v1.13.z/tensorflow_v1.13.z")
set(EXTRA_PREFIX
    "/Users/taozj/Dropbox/repos/macOS")

set(INCLUDE_DIRS
    ${TENSORFLOW_PREFIX}/
    ${TENSORFLOW_PREFIX}/bazel-genfiles
    ${EXTRA_PREFIX}/include/
    ${EXTRA_PREFIX}/include/eigen3/
)
include_directories( ${INCLUDE_DIRS} )

link_directories(
    ${TENSORFLOW_PREFIX}/bazel-bin/tensorflow/
)


set (EXTRA_LIBS ${EXTRA_LIBS} tensorflow_cc tensorflow_framework)
add_executable(start_tf_mnist mnist.cc)
target_link_libraries(
    start_tf_mnist
    ${EXTRA_LIBS}
)