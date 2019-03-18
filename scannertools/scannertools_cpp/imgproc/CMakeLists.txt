set(NEW_SOURCES
  imgproc/blur_kernel_cpu.cpp
  imgproc/histogram_kernel_cpu.cpp
  imgproc/montage_kernel_cpu.cpp
  imgproc/image_encoder_kernel_cpu.cpp
  imgproc/image_decoder_kernel_cpu.cpp
  imgproc/resize_kernel.cpp
  imgproc/feature_extractor_kernel.cpp
  imgproc/optical_flow_kernel_cpu.cpp)

if (BUILD_CUDA)
  list(APPEND NEW_SOURCES
    imgproc/histogram_kernel_gpu.cpp
    imgproc/montage_kernel_gpu.cpp
    imgproc/feature_matcher_kernel.cpp
    imgproc/optical_flow_kernel_gpu.cpp)
    #image_decoder_kernel_gpu.cpp
endif()


SET(SOURCES ${SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
