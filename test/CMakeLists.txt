

add_executable(trt_engine_test test_trt_engine.cc)
target_link_libraries(trt_engine_test trt_engine gtest pthread)
# add_test(NAME test_trt_engine COMMAND trt_engine_test)
include_directories(${CMAKE_SOURCE_DIR})
# trt_test_tiny(test_trt_engine SRCS test_)

