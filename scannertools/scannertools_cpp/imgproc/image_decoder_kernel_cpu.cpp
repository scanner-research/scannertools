#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "scannertools_imgproc.pb.h"
#include "scanner/util/thread_pool.h"

namespace scanner {

class ImageDecoderKernelCPU : public BatchedKernel {
 public:
  ImageDecoderKernelCPU(const KernelConfig& config) : BatchedKernel(config), pool_(32) {}

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    i32 input_count = num_rows(input_columns[0]);

    auto decode = [&](i32 i) {
      std::vector<u8> input_buf(
        input_columns[0][i].buffer,
        input_columns[0][i].buffer + input_columns[0][i].size);
      cv::Mat img = cv::imdecode(input_buf, cv::IMREAD_UNCHANGED);
      if (img.channels() == 4) {
        cv::cvtColor(img, img, CV_BGRA2RGBA);
      } else if (img.channels() == 3) {
        cv::cvtColor(img, img, CV_BGR2RGB);
      }
      LOG_IF(FATAL, img.empty() || !img.data) << "Failed to decode image";
      return img;
    };

    std::vector<std::future<cv::Mat>> futures;
    for (i32 i = 0; i < input_count; ++i) {
      futures.push_back(pool_.enqueue(decode, i));
    }

    std::vector<cv::Mat> images;
    for (i32 i = 0; i < input_count; ++i) {
      images.push_back(futures[i].get());
    }

    std::vector<Frame*> output_frames = new_frames(CPU_DEVICE, mat_to_frame_info(images[0]), input_count);
    for (i32 i = 0; i < input_count; ++i) {
      size_t size = images[i].total() * images[i].elemSize();
      std::memcpy(output_frames[i]->data, images[i].data, size);
      insert_frame(output_columns[0], output_frames[i]);
    }
  }

  ThreadPool pool_;
};

REGISTER_OP(ImageDecoder).input("img").frame_output("frame");

REGISTER_KERNEL(ImageDecoder, ImageDecoderKernelCPU)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(1);
}
