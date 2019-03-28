set(SOURCES
  packed_file_source.cpp
  python_source.cpp
  files_source.cpp
  files_sink.cpp
  captions_source.cpp
  audio_source.cpp)

build_op(
  LIB_NAME scannertools_storage
  CPP_SRCS ${SOURCES}
  PROTO_SRC scannertools_storage.proto)

set(PYBIND11_PYTHON_VERSION 3)
find_package(pybind11 REQUIRED)
target_include_directories(scannertools_storage PUBLIC "${PYTHON_INCLUDE_DIRS}")

target_compile_definitions(scannertools_storage PUBLIC -DHAVE_FFMPEG)
