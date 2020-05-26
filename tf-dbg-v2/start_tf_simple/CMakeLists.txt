cmake_minimum_required (VERSION 2.8.12)
project(start_tf_simple)

include("${PROJECT_SOURCE_DIR}/help.cmake")
define_colors()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wno-shadow -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)

option(BUILD_DEBUG "build project for debug or development propurse..." ON)

# 编译结果中的库和可执行文件的路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)



set(TENSORFLOW_PREFIX
    "/Users/taozj/Dropbox/repos/machine_learning/TENSORFLOW/tensorflow-branch-master/tensorflow-master")
set(EXTRA_PREFIX
    "/Users/taozj/Dropbox/repos/macOS")

set(INCLUDE_DIRS
    ${TENSORFLOW_PREFIX}/
    ${TENSORFLOW_PREFIX}/bazel-bin
    ${EXTRA_PREFIX}/tensorflow-master/include
    ${EXTRA_PREFIX}/include/
    ${EXTRA_PREFIX}/include/eigen3/
)
include_directories( ${INCLUDE_DIRS} )

link_directories(
    ${TENSORFLOW_PREFIX}/bazel-bin/tensorflow
    ${EXTRA_PREFIX}/lib
)

set (EXTRA_LIBS ${EXTRA_LIBS} tensorflow_cc)
set (EXTRA_LIBS ${EXTRA_LIBS} glog)
set (EXTRA_LIBS ${EXTRA_LIBS} pthread)

# saved_model version of add
add_executable(start_tf_saved_add saved_add.cc)
target_link_libraries(
    start_tf_saved_add
    ${EXTRA_LIBS} protobuf
)
