#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include <vector>

namespace scanner {
namespace {
const i32 BINS = 64;
}

class FlowHistogramKernelCPU : public BatchedKernel {
 public:
  FlowHistogramKernelCPU(const KernelConfig& config)
    : BatchedKernel(config), device_(config.devices[0]) {}

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    auto& frame_col = input_columns[0];

    size_t hist_size = BINS * 2 * sizeof(int);
    i32 input_count = num_rows(frame_col);

    u8* output_block = new_block_buffer_size(device_, hist_size, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img = frame_to_mat(frame_col[i].as_const_frame());
      std::vector<cv::Mat> xy_planes;
      split( img, xy_planes );

      cv::Mat deg, mag;

      cv::cartToPolar(xy_planes[0], xy_planes[1], mag, deg, true);

      float magRange[] = {0, 64.0};
      const float* magHistRange = {magRange};

      float degRange[] = {0, 360};
      const float* degHistRange = {degRange};

      u8* output_buf = output_block + i * hist_size;

      for (i32 j = 0; j < 2; ++j) {
        int channels[] = {0};
        cv::Mat hist;
        cv::calcHist(j == 0 ? &mag : &deg, 1, channels, cv::Mat(),
                     hist,
                     1, &BINS,
                     j == 0 ? &magHistRange : & degHistRange);
        cv::Mat out(BINS, 1, CV_32SC1, output_buf + j * BINS * sizeof(int));
        hist.convertTo(out, CV_32SC1);
      }

      insert_element(output_columns[0], output_buf, hist_size);
    }
  }

 private:
  DeviceHandle device_;
};

REGISTER_OP(FlowHistogram).frame_input("flow").output("histogram");

REGISTER_KERNEL(FlowHistogram, FlowHistogramKernelCPU)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(1);
}
