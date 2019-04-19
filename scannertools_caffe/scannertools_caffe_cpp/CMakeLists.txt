cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

execute_process(
  OUTPUT_VARIABLE SCANNER_CMAKE_PATH
  COMMAND python3 -c "import scannerpy.build_flags as b; b.print_cmake()")
include(${SCANNER_CMAKE_PATH})

set(SOURCES
  caffe_kernel.cpp
  caffe_kernel_cpu.cpp
  # Removed until halide '_halide_downgrade_buffer_t' symbol issue resolved
  # caffe_input_kernel.cpp
  # caffe_input_kernel_cpu.cpp
  facenet_input_kernel_cpu.cpp
  facenet_kernel.cpp
  facenet_output_kernel_cpu.cpp
  yolo_output_kernel_cpu.cpp
  faster_rcnn_kernel.cpp
  faster_rcnn_output_kernel_cpu.cpp
  openpose_kernel.cpp)

if (BUILD_CUDA)
  list(APPEND SOURCES
    caffe_kernel_gpu.cpp
    # Removed until halide '_halide_downgrade_buffer_t' symbol issue resolved
    #caffe_input_kernel_gpu.cpp
    # Removed until OpenCV4 GPU issue is resolved
    #facenet_input_kernel_gpu.cpp
    )
endif()

build_op(
  LIB_NAME scannertools_caffe
  CPP_SRCS ${SOURCES}
  PROTO_SRC scannertools_caffe.proto
  BUILD_CUDA ${BUILD_CUDA})

set(LIBRARIES)
set(INCLUDES)
set(DEFS)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

#halide_library(caffe_input_transformer_cpu
#  SRCS caffe_input_transformer_cpu.cpp)
#add_dependencies(caffe caffe_input_transformer_cpu)
#include_directories(${CMAKE_BINARY_DIR}/genfiles/caffe_input_transformer_cpu)
if (BUILD_CUDA)
  # halide_library(caffe_input_transformer_gpu
  #   SRCS caffe_input_transformer_gpu.cpp
  #   HALIDE_TARGET cuda)
  # include_directories(${CMAKE_BINARY_DIR}/genfiles/caffe_input_transformer_gpu)
  # add_dependencies(caffe caffe_input_transformer_gpu)
else()
  # For Caffe
  list(APPEND DEFS -DCPU_ONLY)

endif()

set(OPENCV_COMPONENTS core highgui imgproc video videoio)
if (BUILD_CUDA)
  list(APPEND OPENCV_COMPONENTS cudaarithm cudawarping cudaimgproc)
endif()
find_package(OpenCV REQUIRED COMPONENTS "${OPENCV_COMPONENTS}")
list(APPEND INCLUDES "${OpenCV_INCLUDE_DIRS}")
list(APPEND LIBRARIES "${OpenCV_LIBRARIES}")
list(APPEND DEFS -DUSE_OPENCV)

find_package(Caffe REQUIRED)
list(APPEND INCLUDES "${CAFFE_INCLUDE_DIRS}")
list(APPEND LIBRARIES "${CAFFE_LIBRARIES}")
list(APPEND DEFS -DHAVE_CAFFE)

set(PYBIND11_PYTHON_VERSION 3)
find_package(pybind11 REQUIRED)
list(APPEND INCLUDES "${PYTHON_INCLUDE_DIRS}")

find_package(OpenPose REQUIRED)
list(APPEND INCLUDES "${OPENPOSE_INCLUDE_DIRS}")
list(APPEND LIBRARIES "${OPENPOSE_LIBRARIES}")

target_link_libraries(scannertools_caffe PUBLIC ${LIBRARIES})
target_include_directories(scannertools_caffe PUBLIC ${INCLUDES})
target_compile_definitions(scannertools_caffe PUBLIC ${DEFS})
