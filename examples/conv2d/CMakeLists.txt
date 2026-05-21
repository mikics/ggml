#
# sam

set(TEST_TARGET conv2d)
add_executable(${TEST_TARGET} conv2d.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

#
# sam-quantize

#set(TEST_TARGET sam-quantize)
#add_executable(${TEST_TARGET} quantize.cpp)
#target_link_libraries(${TEST_TARGET} PRIVATE ggml common)
