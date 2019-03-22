#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "scannertools_imgproc.pb.h"

namespace scanner {
namespace {
const std::map<std::string, int> COLOR_CONVERSION_TYPES =
    {
        {u8"COLOR_BGR2BGRA", cv::COLOR_BGR2BGRA},
        {u8"COLOR_RGB2RGBA", cv::COLOR_RGB2RGBA},
        {u8"COLOR_BGRA2BGR", cv::COLOR_BGRA2BGR},
        {u8"COLOR_RGBA2RGB", cv::COLOR_RGBA2RGB},
        {u8"COLOR_BGR2RGBA", cv::COLOR_BGR2RGBA},
        {u8"COLOR_RGB2BGRA", cv::COLOR_RGB2BGRA},
        {u8"COLOR_RGBA2BGR", cv::COLOR_RGBA2BGR},
        {u8"COLOR_BGRA2RGB", cv::COLOR_BGRA2RGB},
        {u8"COLOR_BGR2RGB", cv::COLOR_BGR2RGB},
        {u8"COLOR_RGB2BGR", cv::COLOR_RGB2BGR},
        {u8"COLOR_BGRA2RGBA", cv::COLOR_BGRA2RGBA},
        {u8"COLOR_RGBA2BGRA", cv::COLOR_RGBA2BGRA},
        {u8"COLOR_BGR2GRAY", cv::COLOR_BGR2GRAY},
        {u8"COLOR_RGB2GRAY", cv::COLOR_RGB2GRAY},
        {u8"COLOR_GRAY2BGR", cv::COLOR_GRAY2BGR},
        {u8"COLOR_GRAY2RGB", cv::COLOR_GRAY2RGB},
        {u8"COLOR_GRAY2BGRA", cv::COLOR_GRAY2BGRA},
        {u8"COLOR_GRAY2RGBA", cv::COLOR_GRAY2RGBA},
        {u8"COLOR_BGRA2GRAY", cv::COLOR_BGRA2GRAY},
        {u8"COLOR_RGBA2GRAY", cv::COLOR_RGBA2GRAY},
        {u8"COLOR_BGR2BGR565", cv::COLOR_BGR2BGR565},
        {u8"COLOR_RGB2BGR565", cv::COLOR_RGB2BGR565},
        {u8"COLOR_BGR5652BGR", cv::COLOR_BGR5652BGR},
        {u8"COLOR_BGR5652RGB", cv::COLOR_BGR5652RGB},
        {u8"COLOR_BGRA2BGR565", cv::COLOR_BGRA2BGR565},
        {u8"COLOR_RGBA2BGR565", cv::COLOR_RGBA2BGR565},
        {u8"COLOR_BGR5652BGRA", cv::COLOR_BGR5652BGRA},
        {u8"COLOR_BGR5652RGBA", cv::COLOR_BGR5652RGBA},
        {u8"COLOR_GRAY2BGR565", cv::COLOR_GRAY2BGR565},
        {u8"COLOR_BGR5652GRAY", cv::COLOR_BGR5652GRAY},
        {u8"COLOR_BGR2BGR555", cv::COLOR_BGR2BGR555},
        {u8"COLOR_RGB2BGR555", cv::COLOR_RGB2BGR555},
        {u8"COLOR_BGR5552BGR", cv::COLOR_BGR5552BGR},
        {u8"COLOR_BGR5552RGB", cv::COLOR_BGR5552RGB},
        {u8"COLOR_BGRA2BGR555", cv::COLOR_BGRA2BGR555},
        {u8"COLOR_RGBA2BGR555", cv::COLOR_RGBA2BGR555},
        {u8"COLOR_BGR5552BGRA", cv::COLOR_BGR5552BGRA},
        {u8"COLOR_BGR5552RGBA", cv::COLOR_BGR5552RGBA},
        {u8"COLOR_GRAY2BGR555", cv::COLOR_GRAY2BGR555},
        {u8"COLOR_BGR5552GRAY", cv::COLOR_BGR5552GRAY},
        {u8"COLOR_BGR2XYZ", cv::COLOR_BGR2XYZ},
        {u8"COLOR_RGB2XYZ", cv::COLOR_RGB2XYZ},
        {u8"COLOR_XYZ2BGR", cv::COLOR_XYZ2BGR},
        {u8"COLOR_XYZ2RGB", cv::COLOR_XYZ2RGB},
        {u8"COLOR_BGR2YCrCb", cv::COLOR_BGR2YCrCb},
        {u8"COLOR_RGB2YCrCb", cv::COLOR_RGB2YCrCb},
        {u8"COLOR_YCrCb2BGR", cv::COLOR_YCrCb2BGR},
        {u8"COLOR_YCrCb2RGB", cv::COLOR_YCrCb2RGB},
        {u8"COLOR_BGR2HSV", cv::COLOR_BGR2HSV},
        {u8"COLOR_RGB2HSV", cv::COLOR_RGB2HSV},
        {u8"COLOR_BGR2Lab", cv::COLOR_BGR2Lab},
        {u8"COLOR_RGB2Lab", cv::COLOR_RGB2Lab},
        {u8"COLOR_BGR2Luv", cv::COLOR_BGR2Luv},
        {u8"COLOR_RGB2Luv", cv::COLOR_RGB2Luv},
        {u8"COLOR_BGR2HLS", cv::COLOR_BGR2HLS},
        {u8"COLOR_RGB2HLS", cv::COLOR_RGB2HLS},
        {u8"COLOR_HSV2BGR", cv::COLOR_HSV2BGR},
        {u8"COLOR_HSV2RGB", cv::COLOR_HSV2RGB},
        {u8"COLOR_Lab2BGR", cv::COLOR_Lab2BGR},
        {u8"COLOR_Lab2RGB", cv::COLOR_Lab2RGB},
        {u8"COLOR_Luv2BGR", cv::COLOR_Luv2BGR},
        {u8"COLOR_Luv2RGB", cv::COLOR_Luv2RGB},
        {u8"COLOR_HLS2BGR", cv::COLOR_HLS2BGR},
        {u8"COLOR_HLS2RGB", cv::COLOR_HLS2RGB},
        {u8"COLOR_BGR2HSV_FULL", cv::COLOR_BGR2HSV_FULL},
        {u8"COLOR_RGB2HSV_FULL", cv::COLOR_RGB2HSV_FULL},
        {u8"COLOR_BGR2HLS_FULL", cv::COLOR_BGR2HLS_FULL},
        {u8"COLOR_RGB2HLS_FULL", cv::COLOR_RGB2HLS_FULL},
        {u8"COLOR_HSV2BGR_FULL", cv::COLOR_HSV2BGR_FULL},
        {u8"COLOR_HSV2RGB_FULL", cv::COLOR_HSV2RGB_FULL},
        {u8"COLOR_HLS2BGR_FULL", cv::COLOR_HLS2BGR_FULL},
        {u8"COLOR_HLS2RGB_FULL", cv::COLOR_HLS2RGB_FULL},
        {u8"COLOR_LBGR2Lab", cv::COLOR_LBGR2Lab},
        {u8"COLOR_LRGB2Lab", cv::COLOR_LRGB2Lab},
        {u8"COLOR_LBGR2Luv", cv::COLOR_LBGR2Luv},
        {u8"COLOR_LRGB2Luv", cv::COLOR_LRGB2Luv},
        {u8"COLOR_Lab2LBGR", cv::COLOR_Lab2LBGR},
        {u8"COLOR_Lab2LRGB", cv::COLOR_Lab2LRGB},
        {u8"COLOR_Luv2LBGR", cv::COLOR_Luv2LBGR},
        {u8"COLOR_Luv2LRGB", cv::COLOR_Luv2LRGB},
        {u8"COLOR_BGR2YUV", cv::COLOR_BGR2YUV},
        {u8"COLOR_RGB2YUV", cv::COLOR_RGB2YUV},
        {u8"COLOR_YUV2BGR", cv::COLOR_YUV2BGR},
        {u8"COLOR_YUV2RGB", cv::COLOR_YUV2RGB},
        {u8"COLOR_YUV2RGB_NV12", cv::COLOR_YUV2RGB_NV12},
        {u8"COLOR_YUV2BGR_NV12", cv::COLOR_YUV2BGR_NV12},
        {u8"COLOR_YUV2RGB_NV21", cv::COLOR_YUV2RGB_NV21},
        {u8"COLOR_YUV2BGR_NV21", cv::COLOR_YUV2BGR_NV21},
        {u8"COLOR_YUV420sp2RGB", cv::COLOR_YUV420sp2RGB},
        {u8"COLOR_YUV420sp2BGR", cv::COLOR_YUV420sp2BGR},
        {u8"COLOR_YUV2RGBA_NV12", cv::COLOR_YUV2RGBA_NV12},
        {u8"COLOR_YUV2BGRA_NV12", cv::COLOR_YUV2BGRA_NV12},
        {u8"COLOR_YUV2RGBA_NV21", cv::COLOR_YUV2RGBA_NV21},
        {u8"COLOR_YUV2BGRA_NV21", cv::COLOR_YUV2BGRA_NV21},
        {u8"COLOR_YUV420sp2RGBA", cv::COLOR_YUV420sp2RGBA},
        {u8"COLOR_YUV420sp2BGRA", cv::COLOR_YUV420sp2BGRA},
        {u8"COLOR_YUV2RGB_YV12", cv::COLOR_YUV2RGB_YV12},
        {u8"COLOR_YUV2BGR_YV12", cv::COLOR_YUV2BGR_YV12},
        {u8"COLOR_YUV2RGB_IYUV", cv::COLOR_YUV2RGB_IYUV},
        {u8"COLOR_YUV2BGR_IYUV", cv::COLOR_YUV2BGR_IYUV},
        {u8"COLOR_YUV2RGB_I420", cv::COLOR_YUV2RGB_I420},
        {u8"COLOR_YUV2BGR_I420", cv::COLOR_YUV2BGR_I420},
        {u8"COLOR_YUV420p2RGB", cv::COLOR_YUV420p2RGB},
        {u8"COLOR_YUV420p2BGR", cv::COLOR_YUV420p2BGR},
        {u8"COLOR_YUV2RGBA_YV12", cv::COLOR_YUV2RGBA_YV12},
        {u8"COLOR_YUV2BGRA_YV12", cv::COLOR_YUV2BGRA_YV12},
        {u8"COLOR_YUV2RGBA_IYUV", cv::COLOR_YUV2RGBA_IYUV},
        {u8"COLOR_YUV2BGRA_IYUV", cv::COLOR_YUV2BGRA_IYUV},
        {u8"COLOR_YUV2RGBA_I420", cv::COLOR_YUV2RGBA_I420},
        {u8"COLOR_YUV2BGRA_I420", cv::COLOR_YUV2BGRA_I420},
        {u8"COLOR_YUV420p2RGBA", cv::COLOR_YUV420p2RGBA},
        {u8"COLOR_YUV420p2BGRA", cv::COLOR_YUV420p2BGRA},
        {u8"COLOR_YUV2GRAY_420", cv::COLOR_YUV2GRAY_420},
        {u8"COLOR_YUV2GRAY_NV21", cv::COLOR_YUV2GRAY_NV21},
        {u8"COLOR_YUV2GRAY_NV12", cv::COLOR_YUV2GRAY_NV12},
        {u8"COLOR_YUV2GRAY_YV12", cv::COLOR_YUV2GRAY_YV12},
        {u8"COLOR_YUV2GRAY_IYUV", cv::COLOR_YUV2GRAY_IYUV},
        {u8"COLOR_YUV2GRAY_I420", cv::COLOR_YUV2GRAY_I420},
        {u8"COLOR_YUV420sp2GRAY", cv::COLOR_YUV420sp2GRAY},
        {u8"COLOR_YUV420p2GRAY", cv::COLOR_YUV420p2GRAY},
        {u8"COLOR_YUV2RGB_UYVY", cv::COLOR_YUV2RGB_UYVY},
        {u8"COLOR_YUV2BGR_UYVY", cv::COLOR_YUV2BGR_UYVY},
        {u8"COLOR_YUV2RGB_Y422", cv::COLOR_YUV2RGB_Y422},
        {u8"COLOR_YUV2BGR_Y422", cv::COLOR_YUV2BGR_Y422},
        {u8"COLOR_YUV2RGB_UYNV", cv::COLOR_YUV2RGB_UYNV},
        {u8"COLOR_YUV2BGR_UYNV", cv::COLOR_YUV2BGR_UYNV},
        {u8"COLOR_YUV2RGBA_UYVY", cv::COLOR_YUV2RGBA_UYVY},
        {u8"COLOR_YUV2BGRA_UYVY", cv::COLOR_YUV2BGRA_UYVY},
        {u8"COLOR_YUV2RGBA_Y422", cv::COLOR_YUV2RGBA_Y422},
        {u8"COLOR_YUV2BGRA_Y422", cv::COLOR_YUV2BGRA_Y422},
        {u8"COLOR_YUV2RGBA_UYNV", cv::COLOR_YUV2RGBA_UYNV},
        {u8"COLOR_YUV2BGRA_UYNV", cv::COLOR_YUV2BGRA_UYNV},
        {u8"COLOR_YUV2RGB_YUY2", cv::COLOR_YUV2RGB_YUY2},
        {u8"COLOR_YUV2BGR_YUY2", cv::COLOR_YUV2BGR_YUY2},
        {u8"COLOR_YUV2RGB_YVYU", cv::COLOR_YUV2RGB_YVYU},
        {u8"COLOR_YUV2BGR_YVYU", cv::COLOR_YUV2BGR_YVYU},
        {u8"COLOR_YUV2RGB_YUYV", cv::COLOR_YUV2RGB_YUYV},
        {u8"COLOR_YUV2BGR_YUYV", cv::COLOR_YUV2BGR_YUYV},
        {u8"COLOR_YUV2RGB_YUNV", cv::COLOR_YUV2RGB_YUNV},
        {u8"COLOR_YUV2BGR_YUNV", cv::COLOR_YUV2BGR_YUNV},
        {u8"COLOR_YUV2RGBA_YUY2", cv::COLOR_YUV2RGBA_YUY2},
        {u8"COLOR_YUV2BGRA_YUY2", cv::COLOR_YUV2BGRA_YUY2},
        {u8"COLOR_YUV2RGBA_YVYU", cv::COLOR_YUV2RGBA_YVYU},
        {u8"COLOR_YUV2BGRA_YVYU", cv::COLOR_YUV2BGRA_YVYU},
        {u8"COLOR_YUV2RGBA_YUYV", cv::COLOR_YUV2RGBA_YUYV},
        {u8"COLOR_YUV2BGRA_YUYV", cv::COLOR_YUV2BGRA_YUYV},
        {u8"COLOR_YUV2RGBA_YUNV", cv::COLOR_YUV2RGBA_YUNV},
        {u8"COLOR_YUV2BGRA_YUNV", cv::COLOR_YUV2BGRA_YUNV},
        {u8"COLOR_YUV2GRAY_UYVY", cv::COLOR_YUV2GRAY_UYVY},
        {u8"COLOR_YUV2GRAY_YUY2", cv::COLOR_YUV2GRAY_YUY2},
        {u8"COLOR_YUV2GRAY_Y422", cv::COLOR_YUV2GRAY_Y422},
        {u8"COLOR_YUV2GRAY_UYNV", cv::COLOR_YUV2GRAY_UYNV},
        {u8"COLOR_YUV2GRAY_YVYU", cv::COLOR_YUV2GRAY_YVYU},
        {u8"COLOR_YUV2GRAY_YUYV", cv::COLOR_YUV2GRAY_YUYV},
        {u8"COLOR_YUV2GRAY_YUNV", cv::COLOR_YUV2GRAY_YUNV},
        {u8"COLOR_RGBA2mRGBA", cv::COLOR_RGBA2mRGBA},
        {u8"COLOR_mRGBA2RGBA", cv::COLOR_mRGBA2RGBA},
        {u8"COLOR_RGB2YUV_I420", cv::COLOR_RGB2YUV_I420},
        {u8"COLOR_BGR2YUV_I420", cv::COLOR_BGR2YUV_I420},
        {u8"COLOR_RGB2YUV_IYUV", cv::COLOR_RGB2YUV_IYUV},
        {u8"COLOR_BGR2YUV_IYUV", cv::COLOR_BGR2YUV_IYUV},
        {u8"COLOR_RGBA2YUV_I420", cv::COLOR_RGBA2YUV_I420},
        {u8"COLOR_BGRA2YUV_I420", cv::COLOR_BGRA2YUV_I420},
        {u8"COLOR_RGBA2YUV_IYUV", cv::COLOR_RGBA2YUV_IYUV},
        {u8"COLOR_BGRA2YUV_IYUV", cv::COLOR_BGRA2YUV_IYUV},
        {u8"COLOR_RGB2YUV_YV12", cv::COLOR_RGB2YUV_YV12},
        {u8"COLOR_BGR2YUV_YV12", cv::COLOR_BGR2YUV_YV12},
        {u8"COLOR_RGBA2YUV_YV12", cv::COLOR_RGBA2YUV_YV12},
        {u8"COLOR_BGRA2YUV_YV12", cv::COLOR_BGRA2YUV_YV12},
        {u8"COLOR_BayerBG2BGR", cv::COLOR_BayerBG2BGR},
        {u8"COLOR_BayerGB2BGR", cv::COLOR_BayerGB2BGR},
        {u8"COLOR_BayerRG2BGR", cv::COLOR_BayerRG2BGR},
        {u8"COLOR_BayerGR2BGR", cv::COLOR_BayerGR2BGR},
        {u8"COLOR_BayerBG2RGB", cv::COLOR_BayerBG2RGB},
        {u8"COLOR_BayerGB2RGB", cv::COLOR_BayerGB2RGB},
        {u8"COLOR_BayerRG2RGB", cv::COLOR_BayerRG2RGB},
        {u8"COLOR_BayerGR2RGB", cv::COLOR_BayerGR2RGB},
        {u8"COLOR_BayerBG2GRAY", cv::COLOR_BayerBG2GRAY},
        {u8"COLOR_BayerGB2GRAY", cv::COLOR_BayerGB2GRAY},
        {u8"COLOR_BayerRG2GRAY", cv::COLOR_BayerRG2GRAY},
        {u8"COLOR_BayerGR2GRAY", cv::COLOR_BayerGR2GRAY},
        {u8"COLOR_BayerBG2BGR_VNG", cv::COLOR_BayerBG2BGR_VNG},
        {u8"COLOR_BayerGB2BGR_VNG", cv::COLOR_BayerGB2BGR_VNG},
        {u8"COLOR_BayerRG2BGR_VNG", cv::COLOR_BayerRG2BGR_VNG},
        {u8"COLOR_BayerGR2BGR_VNG", cv::COLOR_BayerGR2BGR_VNG},
        {u8"COLOR_BayerBG2RGB_VNG", cv::COLOR_BayerBG2RGB_VNG},
        {u8"COLOR_BayerGB2RGB_VNG", cv::COLOR_BayerGB2RGB_VNG},
        {u8"COLOR_BayerRG2RGB_VNG", cv::COLOR_BayerRG2RGB_VNG},
        {u8"COLOR_BayerGR2RGB_VNG", cv::COLOR_BayerGR2RGB_VNG},
        {u8"COLOR_BayerBG2BGR_EA", cv::COLOR_BayerBG2BGR_EA},
        {u8"COLOR_BayerGB2BGR_EA", cv::COLOR_BayerGB2BGR_EA},
        {u8"COLOR_BayerRG2BGR_EA", cv::COLOR_BayerRG2BGR_EA},
        {u8"COLOR_BayerGR2BGR_EA", cv::COLOR_BayerGR2BGR_EA},
        {u8"COLOR_BayerBG2RGB_EA", cv::COLOR_BayerBG2RGB_EA},
        {u8"COLOR_BayerGB2RGB_EA", cv::COLOR_BayerGB2RGB_EA},
        {u8"COLOR_BayerRG2RGB_EA", cv::COLOR_BayerRG2RGB_EA},
        {u8"COLOR_BayerGR2RGB_EA", cv::COLOR_BayerGR2RGB_EA},
        {u8"COLOR_COLORCVT_MAX", cv::COLOR_COLORCVT_MAX},
    };
}

class ConvertColorKernel : public BatchedKernel {
public:
  ConvertColorKernel(const KernelConfig& config)
    : BatchedKernel(config), device_(config.devices[0]) {
    valid_.set_success(true);
  }

  void new_stream(const std::vector<u8>& args) override {
    args_.ParseFromArray(args.data(), args.size());

    color_convert_type_ = cv::INTER_LINEAR;
    int new_color_type;
    if (COLOR_CONVERSION_TYPES.count(args_.conversion()) > 0) {
      new_color_type = COLOR_CONVERSION_TYPES.at(args_.conversion());
    } else {
      std::string err =
          "ConvertColor: invalid color conversion argument provided: " +
          args_.conversion();
      RESULT_ERROR(&valid_, err.c_str());
    }
    if (new_color_type != color_convert_type_) {
      output_channel_count_ = 0;
    }
    color_convert_type_ = new_color_type;
  }

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    auto& frame_col = input_columns[0];
    set_device();

    const Frame* frame = frame_col[0].as_const_frame();

    if (output_channel_count_ == 0) {
      if (device_.type == DeviceType::CPU) {
        cv::Mat output;
        cv::cvtColor(frame_to_mat(frame), output, color_convert_type_);
        output_channel_count_ = output.channels();
        output_channel_type_ = cv_to_frame_type(output.depth());
      } else {
        CUDA_PROTECT({
          cvc::GpuMat output;
          cv::cvtColor(frame_to_gpu_mat(frame), output, color_convert_type_);
          output_channel_count_ = output.channels();
          output_channel_type_ = cv_to_frame_type(output.depth());
        });
      }
    }

    i32 input_count = num_rows(frame_col);
    FrameInfo info(frame->height(), frame->width(), output_channel_count_,
                   output_channel_type_);
    std::vector<Frame *> output_frames = new_frames(device_, info, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      if (device_.type == DeviceType::CPU) {
        cv::Mat img = frame_to_mat(frame_col[i].as_const_frame());
        cv::Mat out_mat = frame_to_mat(output_frames[i]);
        cv::cvtColor(img, out_mat, color_convert_type_);
      } else {
        CUDA_PROTECT({
          cvc::GpuMat img = frame_to_gpu_mat(frame_col[i].as_const_frame());
          cvc::GpuMat out_mat = frame_to_gpu_mat(output_frames[i]);
          cvc::cvtColor(img, out_mat, color_convert_type_);
        });
      }
      insert_frame(output_columns[0], output_frames[i]);
    }
  }

  void set_device() {
    if (device_.type == DeviceType::GPU) {
      CUDA_PROTECT({
        CU_CHECK(cudaSetDevice(device_.id));
        cvc::setDevice(device_.id);
      });
    }
  }

 private:
  Result valid_;
  DeviceHandle device_;
  ConvertColorArgs args_;
  int color_convert_type_;
  int output_channel_count_ = 0;
  FrameType output_channel_type_;
};

REGISTER_OP(ConvertColor).frame_input("frame").frame_output("frame").stream_protobuf_name(
    "ConvertColorArgs");

REGISTER_KERNEL(ConvertColor, ConvertColorKernel).device(DeviceType::CPU).num_devices(1);

#ifdef HAVE_CUDA
REGISTER_KERNEL(ConvertColor, ConvertColorKernel).device(DeviceType::GPU).num_devices(1);
#endif
}
