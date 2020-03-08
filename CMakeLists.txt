cmake_minimum_required (VERSION 2.8.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wno-shadow -Wno-sign-compare")
set(CMAKE_CXX_STANDARD 11)

project(start_tf)

# 编译结果中的库和可执行文件的路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

set(DEPENDENCY_PREFIX "../../opensource/")
set(INCLUDE_DIRS
    ${DEPENDENCY_PREFIX}/tensorflow_and_paddle/tensorflow_install_v1.13.1/include
    ${DEPENDENCY_PREFIX}/tensorflow_and_paddle/tensorflow_install_v1.13.1/include/abseil-cpp
    ${DEPENDENCY_PREFIX}/tensorflow_and_paddle/tensorflow_install_v1.13.1/include/eigen_archive
    ${DEPENDENCY_PREFIX}/xtra_rhel7.x/google_prefix/include/
    ${DEPENDENCY_PREFIX}/roo/
)
include_directories( ${INCLUDE_DIRS} )

link_directories(
#        ${DEPENDENCY_PREFIX}/tensorflow_and_paddle/tensorflow_install_v1.13.1/lib_monolithic/
#        ${DEPENDENCY_PREFIX}/tensorflow_and_paddle/tensorflow_install_v1.13.1/lib/
    /home/user/repos/machine_learning/tensorflow_v1.13.z/bazel-bin/tensorflow
)


set (EXTRA_LIBS ${EXTRA_LIBS} tensorflow_cc)

add_executable(start_tf_mnist mnist.cc)
target_link_libraries(
    start_tf_mnist
    ${EXTRA_LIBS}
)

add_executable(start_tf_add add.cc)
target_link_libraries(
    start_tf_add
    ${EXTRA_LIBS}
)

add_executable(start_tf_tensor tensor.cc)
target_link_libraries(
    start_tf_tensor
    ${EXTRA_LIBS}
)
