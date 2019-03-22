cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

execute_process(
  OUTPUT_VARIABLE SCANNER_CMAKE_PATH
  COMMAND python3 -c "import scannerpy.build_flags as b; b.print_cmake()")
include(${SCANNER_CMAKE_PATH})

set(SOURCES tfcpp.cpp)

build_op(
  LIB_NAME scannertools_tfcpp
  CPP_SRCS ${SOURCES})
  #PROTO_SRC scannertools_tfcpp.proto)

set(LIBTF_PATH /home/will/libtensorflow)

set(LIBRARIES ${LIBTF_PATH}/lib/libtensorflow.so)
set(INCLUDES ${LIBTF_PATH}/include)

target_include_directories(scannertools_tfcpp PUBLIC ${INCLUDES})
target_link_libraries(scannertools_tfcpp PUBLIC ${LIBRARIES})
