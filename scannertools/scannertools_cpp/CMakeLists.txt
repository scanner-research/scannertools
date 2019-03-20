cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

execute_process(
  OUTPUT_VARIABLE SCANNER_CMAKE_PATH
  COMMAND python3 -c "import scannerpy.build_flags as b; b.print_cmake()")
include(${SCANNER_CMAKE_PATH})

set(SOURCES)
add_subdirectory(imgproc)
add_subdirectory(storage)
add_subdirectory(misc)
