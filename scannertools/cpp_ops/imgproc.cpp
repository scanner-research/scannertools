#include "imgproc.pb.h"
#include "scanner/api/kernel.h"   // for VideoKernel and REGISTER_KERNEL
#include "scanner/api/op.h"       // for REGISTER_OP
#include "scanner/util/memory.h"  // for device-independent memory management
#include "scanner/util/opencv.h"  // for using OpenCV
#include "scanner/util/serialize.h"

#include <vector>
#include <math.h>
#include <glog/logging.h>

namespace scanner{

class ConvertToHSVKernel : public scanner::Kernel, public scanner::VideoKernel {
  public:
    ConvertToHSVKernel(const scanner::KernelConfig& config)
        : scanner::Kernel(config) {
      ImgProcArgs args;
      args.ParseFromArray(config.args.data(), config.args.size());
      width_ = args.width();
      height_ = args.height();
    }

    void execute(const scanner::Elements& input_columns,
        scanner::Elements& output_columns) override {
      auto& frame_col = input_columns[0];

      check_frame(scanner::CPU_DEVICE, frame_col);

      const scanner::Frame* frame = frame_col.as_const_frame();
      cv::Mat input = scanner::frame_to_mat(frame);

      auto& output_frame_col = output_columns[0];
      scanner::FrameInfo output_frame_info(height_, width_, 3, scanner::FrameType::U8);
      
      scanner::Frame* output_frame =
        scanner::new_frame(scanner::CPU_DEVICE, output_frame_info);
      cv::Mat output = scanner::frame_to_mat(output_frame);

      cv::cvtColor(input, output, cv::COLOR_RGB2HSV);

      insert_frame(output_frame_col, output_frame);
    }

  private:
    int width_;
    int height_;
};

class BrightnessKernel : public scanner::Kernel, public scanner::VideoKernel {
  public:
    BrightnessKernel(const scanner::KernelConfig& config)
        : scanner::Kernel(config) {
      ImgProcArgs args;
      args.ParseFromArray(config.args.data(), config.args.size());
      width_ = args.width();
      height_ = args.height();
    }

    void execute(const scanner::Elements& input_columns,
        scanner::Elements& output_columns) override {
      auto& frame_col = input_columns[0];

      check_frame(scanner::CPU_DEVICE, frame_col);
      
      u8* output_buf = new_buffer(scanner::CPU_DEVICE, sizeof(float));
      
      const scanner::Frame* frame = frame_col.as_const_frame();
      cv::Mat input = scanner::frame_to_mat(frame);
      cv::Mat output = cv::Mat::zeros(input.size(), input.type());

      cv::cvtColor(input, output, cv::COLOR_RGB2YUV);

      cv::Scalar mean_vals = cv::mean(output);
      float brightness = mean_vals.val[0];

      ((float*)output_buf)[0] = brightness;

      insert_element(output_columns[0], output_buf, sizeof(float));
    }

  private:
    int width_;
    int height_;
};

class ContrastKernel : public scanner::Kernel, public scanner::VideoKernel {
  public:
    ContrastKernel(const scanner::KernelConfig& config)
        : scanner::Kernel(config) {
      ImgProcArgs args;
      args.ParseFromArray(config.args.data(), config.args.size());
      width_ = args.width();
      height_ = args.height();
    }

    void execute(const scanner::Elements& input_columns,
        scanner::Elements& output_columns) override {
      auto& frame_col = input_columns[0];

      check_frame(scanner::CPU_DEVICE, frame_col);
      
      u8* output_buf = new_buffer(scanner::CPU_DEVICE, sizeof(float));
      
      const scanner::Frame* frame = frame_col.as_const_frame();
      cv::Mat input = scanner::frame_to_mat(frame);
      cv::Mat yuv;

      cv::cvtColor(input, yuv, cv::COLOR_RGB2YUV);
     
      cv::Scalar mean_vals = cv::mean(yuv);
      
      float square_error = 0.f;
      for (int row = 0; row < yuv.rows; row++) {
        for (int col = 0; col < yuv.cols; col++) {
          square_error += ((float)yuv.at<u8>(row, col, 0) - (float)mean_vals.val[0]) *
            ((float)yuv.at<u8>(row, col, 0) - (float)mean_vals.val[0]);
        }
      }

      float mean_square_error = square_error / (yuv.rows * yuv.cols);

      float contrast = sqrt(mean_square_error);

      ((float*)output_buf)[0] = contrast;

      insert_element(output_columns[0], output_buf, sizeof(float));
    }

  private:
    int width_;
    int height_;
};

class SharpnessKernel : public scanner::Kernel, public scanner::VideoKernel {
  public:
    SharpnessKernel(const scanner::KernelConfig& config)
        : scanner::Kernel(config) {
      ImgProcArgs args;
      args.ParseFromArray(config.args.data(), config.args.size());
      width_ = args.width();
      height_ = args.height();
    }

    void execute(const scanner::Elements& input_columns,
        scanner::Elements& output_columns) override {
      auto& frame_col = input_columns[0];

      check_frame(scanner::CPU_DEVICE, frame_col);
      
      u8* output_buf = new_buffer(scanner::CPU_DEVICE, sizeof(float));
      
      const scanner::Frame* frame = frame_col.as_const_frame();
      cv::Mat input = scanner::frame_to_mat(frame);
      cv::Mat laplacian;
      
      cv::Laplacian(input, laplacian, CV_64F);

      cv::Scalar mean;
      cv::Scalar stddev;
      cv::meanStdDev(laplacian, mean, stddev);

      cv::pow(stddev, 2, stddev);

      float sharpness = (stddev[0] + stddev[1] + stddev[2]) / 3.f;

      ((float*)output_buf)[0] = sharpness;

      insert_element(output_columns[0], output_buf, sizeof(float));
    }

  private:
    int width_;
    int height_;
};

class SharpnessBBoxKernel : public scanner::Kernel, public scanner::VideoKernel {
  public:
    SharpnessBBoxKernel(const scanner::KernelConfig& config)
        : scanner::Kernel(config) {
      ImgProcArgs args;
      args.ParseFromArray(config.args.data(), config.args.size());
      width_ = args.width();
      height_ = args.height();
    }

    void execute(const scanner::Elements& input_columns,
        scanner::Elements& output_columns) override {
      auto& frame_col = input_columns[0];
      auto& bbox_col = input_columns[1];

      check_frame(scanner::CPU_DEVICE, frame_col);

      const scanner::Frame* frame = frame_col.as_const_frame();
      cv::Mat input = scanner::frame_to_mat(frame);

      std::vector<BoundingBox> all_bboxes =
        deserialize_proto_vector<BoundingBox>(bbox_col.buffer, bbox_col.size);

      u8* output_buf = new_buffer(scanner::CPU_DEVICE,
          all_bboxes.size() * sizeof(float));

      for (int i = 0; i < all_bboxes.size(); i++) {
        auto& bbox = all_bboxes[i];

        int x1 = (int) bbox.x1(), y1 = (int) bbox.y1(),
            x2 = (int) bbox.x2(), y2 = (int) bbox.y2();
        cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        cv::Mat region = input(roi);
        cv::resize(region, region, cv::Size(200, 200));

        cv::Mat laplacian;
        
        cv::Laplacian(region, laplacian, CV_64F);

        cv::Scalar mean;
        cv::Scalar stddev;
        cv::meanStdDev(laplacian, mean, stddev);

        cv::pow(stddev, 2, stddev);

        float sharpness = (stddev[0] + stddev[1] + stddev[2]) / 3.f;

        ((float*)output_buf)[i] = sharpness;
      }

      insert_element(output_columns[0], output_buf,
          all_bboxes.size() * sizeof(float));
    }

  private:
    int width_;
    int height_;
};

REGISTER_OP(ConvertToHSVCPP)
  .frame_input("frame")
  .frame_output("frame")
  .protobuf_name("ImgProcArgs");

REGISTER_KERNEL(ConvertToHSVCPP, ConvertToHSVKernel)
  .device(scanner::DeviceType::CPU)
  .num_devices(1);

REGISTER_OP(BrightnessCPP)
  .frame_input("frame")
  .output("brightness")
  .protobuf_name("ImgProcArgs");

REGISTER_KERNEL(BrightnessCPP, BrightnessKernel)
  .device(scanner::DeviceType::CPU)
  .num_devices(1);

REGISTER_OP(ContrastCPP)
  .frame_input("frame")
  .output("contrast")
  .protobuf_name("ImgProcArgs");

REGISTER_KERNEL(ContrastCPP, ContrastKernel)
  .device(scanner::DeviceType::CPU)
  .num_devices(1);

REGISTER_OP(SharpnessCPP)
  .frame_input("frame")
  .output("sharpness")
  .protobuf_name("ImgProcArgs");

REGISTER_KERNEL(SharpnessCPP, SharpnessKernel)
  .device(scanner::DeviceType::CPU)
  .num_devices(1);

REGISTER_OP(SharpnessBBoxCPP)
  .frame_input("frame")
  .input("bboxes")
  .output("sharpness_bbox")
  .protobuf_name("ImgProcArgs");

REGISTER_KERNEL(SharpnessBBoxCPP, SharpnessBBoxKernel)
  .device(scanner::DeviceType::CPU)
  .num_devices(1);
}

