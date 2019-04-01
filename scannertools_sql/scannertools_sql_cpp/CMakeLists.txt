cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

execute_process(
  OUTPUT_VARIABLE SCANNER_CMAKE_PATH
  COMMAND python3 -c "import scannerpy.build_flags as b; b.print_cmake()")
include(${SCANNER_CMAKE_PATH})

set(SOURCES sql.cpp sql_sink.cpp sql_source.cpp)

build_op(
  LIB_NAME scannertools_sql
  CPP_SRCS ${SOURCES}
  PROTO_SRC scannertools_sql.proto)

set(LIBRARIES)

find_package(PkgConfig REQUIRED)
pkg_check_modules(libpqxx REQUIRED libpqxx)
target_include_directories(scannertools_sql PUBLIC ${libpqxx_INCLUDE_DIRS})
target_link_libraries(scannertools_sql PUBLIC "${libpqxx_LIBRARY_DIRS}/libpqxx.a" pq)
