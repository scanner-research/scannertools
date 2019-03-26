cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

execute_process(
  OUTPUT_VARIABLE SCANNER_CMAKE_PATH
  COMMAND python3 -c "import scannerpy.build_flags as b; b.print_cmake()")
include(${SCANNER_CMAKE_PATH})

set(SOURCES pytorch.cpp)

build_op(
  LIB_NAME scannertools_pytorch
  CPP_SRCS ${SOURCES}
  BUILD_CUDA ${BUILD_CUDA})
  #PROTO_SRC scannertools_pytorch.proto)

execute_process(
  OUTPUT_VARIABLE PYTORCH_PATH
  COMMAND python3 -c "import torch, os, sys; sys.stdout.write(os.path.dirname(torch.__file__))")
find_package(Torch REQUIRED PATHS "${PYTORCH_PATH}/share/cmake/Torch/")

#set(LIBTF_PATH /home/will/libtensorflow)

set(LIBRARIES ${TORCH_LIBRARIES})
set(INCLUDES ${TORCH_INCLUDE_DIRS})

target_include_directories(scannertools_pytorch PUBLIC ${INCLUDES})
target_link_libraries(scannertools_pytorch PUBLIC ${LIBRARIES})
set_property(TARGET scannertools_pytorch PROPERTY CXX_STANDARD 11)
