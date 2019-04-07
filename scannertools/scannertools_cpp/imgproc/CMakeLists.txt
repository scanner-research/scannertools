set(SOURCES
  blur_kernel_cpu.cpp
  histogram_kernel_cpu.cpp
  montage_kernel_cpu.cpp
  image_decoder_kernel_cpu.cpp
  resize_kernel.cpp
  convert_color_kernel.cpp
  optical_flow_kernel_cpu.cpp)

if (BUILD_CUDA)
  list(APPEND SOURCES
    histogram_kernel_gpu.cpp
    montage_kernel_gpu.cpp
    optical_flow_kernel_gpu.cpp)
    #image_decoder_kernel_gpu.cpp
endif()

build_op(
  LIB_NAME scannertools_imgproc
  CPP_SRCS ${SOURCES}
  PROTO_SRC scannertools_imgproc.proto
  BUILD_CUDA ${BUILD_CUDA})

set(LIBRARIES)

list(APPEND OPENCV_COMPONENTS core highgui imgproc optflow)
if (BUILD_CUDA)
  list(APPEND OPENCV_COMPONENTS cudafeatures2d cudacodec cudaoptflow)
endif()

find_package(OpenCV REQUIRED COMPONENTS "${OPENCV_COMPONENTS}")
list(APPEND LIBRARIES "${OpenCV_LIBRARIES}")

target_link_libraries(scannertools_imgproc PUBLIC ${LIBRARIES})
