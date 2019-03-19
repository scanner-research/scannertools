cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

execute_process(
  OUTPUT_VARIABLE SCANNER_CMAKE_PATH
  COMMAND python3 -c "import scannerpy.build_flags as b; b.print_cmake()")
include(${SCANNER_CMAKE_PATH})

set(SOURCES)
add_subdirectory(imgproc)
add_subdirectory(io)
add_subdirectory(misc)

build_op(
  LIB_NAME scannertools
  CPP_SRCS ${SOURCES}
  PROTO_SRC scannertools.proto
  BUILD_CUDA ${BUILD_CUDA})

set(LIBRARIES)

list(APPEND OPENCV_COMPONENTS core highgui imgproc xfeatures2d cudafeatures2d cudacodec)
find_package(OpenCV REQUIRED COMPONENTS "${OPENCV_COMPONENTS}")
list(APPEND LIBRARIES "${OpenCV_LIBRARIES}")

set(PYBIND11_PYTHON_VERSION 3)
find_package(pybind11 REQUIRED)
include_directories("${PYTHON_INCLUDE_DIRS}")

target_link_libraries(scannertools PUBLIC ${LIBRARIES})

add_definitions(-DHAVE_FFMPEG)
