


# 此处增加单元测试用例，会自动查找 xxxx_test.cc
# 额外的单元测试链接库请作为后面额外的参数传入


add_executable(add_test add_test.cc
               ../add.cc)
target_link_libraries(add_test ${GOOGLE_TEST_LIBS} glog tensorflow_cc)
add_test(NAME gtest_add_test COMMAND add_test)