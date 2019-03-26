set(SOURCES
  info_from_frame_kernel.cpp
  sleep_kernel.cpp
  pass_kernel.cpp
  discard_kernel.cpp)

build_op(
  LIB_NAME scannertools_misc
  CPP_SRCS ${SOURCES})
