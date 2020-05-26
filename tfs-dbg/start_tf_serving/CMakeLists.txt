cmake_minimum_required (VERSION 2.8.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wno-shadow -Wno-sign-compare")
set(CMAKE_CXX_STANDARD 11)

project(start_tf_serving)

# 编译结果中的库和可执行文件的路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

set(TENSORFLOW_PREFIX
    "/Users/taozj/Dropbox/repos/machine_learning/TENSORFLOW/tensorflow-branch-v1.13.z/tensorflow_v1.13.z-lite")
set(TENSORFLOW_SERVING_PREFIX
    "/Users/taozj/Dropbox/repos/machine_learning/TENSORFLOW/tensorflow-branch-v1.13.z/serving_v1.13.z-lite")    
set(EXTRA_PREFIX
    "/Users/taozj/Dropbox/repos/macOS")

set(INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${TENSORFLOW_PREFIX}/
    ${TENSORFLOW_PREFIX}/bazel-genfiles
    ${TENSORFLOW_SERVING_PREFIX}/bazel-genfiles
    ${EXTRA_PREFIX}/include/
    ${EXTRA_PREFIX}/include/eigen3/
)
include_directories( ${INCLUDE_DIRS} )

add_subdirectory( apis/ )

link_directories(
    ${TENSORFLOW_PREFIX}/bazel-bin/tensorflow/
    ${EXTRA_PREFIX}/lib
)

set (EXTRA_LIBS ${EXTRA_LIBS} apis)

set (EXTRA_LIBS ${EXTRA_LIBS} protoc protobuf)
set (EXTRA_LIBS ${EXTRA_LIBS} grpc++ grpc gpr)
set (EXTRA_LIBS ${EXTRA_LIBS} pthread z)
set (EXTRA_LIBS ${EXTRA_LIBS} pthread tensorflow_cc)

add_executable(start_tf_client client.cc)
target_link_libraries(
    start_tf_client
    ${EXTRA_LIBS} glog
)
