#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <openpose/headers.hpp>

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/opencv.h"
#include "scanner/util/fs.h"
#include "scannertools_caffe.pb.h"

namespace scanner {

const int POSE_KEYPOINTS = 18;
const int FACE_KEYPOINTS = 70;
const int HAND_KEYPOINTS = 21;
const int POSE_SCORES = 1;
const int TOTAL_KEYPOINTS =
    POSE_KEYPOINTS + FACE_KEYPOINTS + 2 * HAND_KEYPOINTS;

class OpenPoseKernel : public scanner::BatchedKernel,
                       public scanner::VideoKernel {
 public:
  OpenPoseKernel(const scanner::KernelConfig& config)
    : scanner::BatchedKernel(config),
      opWrapper_{op::ThreadManagerMode::Asynchronous},
      device_(config.devices[0]) {
    args_.ParseFromArray(config.args.data(), config.args.size());

    LOG_IF(FATAL, device_.type == DeviceType::CPU)
      << "OpenPose CPU seems to not work correctly right now. Please use the GPU version or contact the "
      << "OpenPose maintainers.";
  }

  void fetch_resources(proto::Result* result) override {
    if (args_.model_directory() == "") {
      std::string model_dir;
      cached_dir("openpose", model_dir);

      // Pose prototxt
      download_if_uncached(
          "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/"
          "openpose/master/models/pose/coco/pose_deploy_linevec.prototxt",
          model_dir + "/pose/coco/pose_deploy_linevec.prototxt");

      // Pose model weights
      const std::string pose_fs_url =
          "http://posefs1.perception.cs.cmu.edu/OpenPose/models";
      download_if_uncached(
            pose_fs_url + "/pose/coco/pose_iter_440000.caffemodel",
            model_dir + "/pose/coco/pose_iter_440000.caffemodel");
      // Hands prototxt
      download_if_uncached(
            "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/"
            "openpose/master/models/hand/pose_deploy.prototxt",
            model_dir + "/hand/pose_deploy.prototxt");
      // Hands model weights
      download_if_uncached(
          pose_fs_url + "/hand/pose_iter_102000.caffemodel",
          model_dir + "/hand/pose_iter_102000.caffemodel");
      // Face prototxt
      download_if_uncached(
          "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/"
          "openpose/master/models/face/pose_deploy.prototxt",
          model_dir + "/face/pose_deploy.prototxt");
      // Face model weights
      download_if_uncached(
          pose_fs_url + "/face/pose_iter_116000.caffemodel",
          model_dir + "/face/pose_iter_116000.caffemodel");
      // Face haar cascades
      download_if_uncached(
          "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/"
          "openpose/master/models/face/haarcascade_frontalface_alt.xml",
          model_dir + "/face/haarcascade_frontalface_alt.xml");
    }

    result->set_success(true);
  }

  void setup_with_resources(proto::Result* result) override {
    std::string model_dir;
    if (args_.model_directory() == "") {
      cached_dir("openpose", model_dir);
    } else {
      model_dir = args_.model_directory();
    }

    int input_resolution = -1;
    if (device_ == CPU_DEVICE) {
      input_resolution = 656;
    }


    const op::PoseModel poseModel = op::PoseModel::COCO_18;

    const op::WrapperStructPose wrapperStructPose{
        op::PoseMode::Enabled,
        {input_resolution, 368},
        {-1, -1},
        op::ScaleMode::ZeroToOne,
        device_.type == DeviceType::GPU ? 1 : 0,
        device_.type == DeviceType::GPU ? device_.id : 0,
        args_.pose_num_scales(),
        args_.pose_scale_gap(),
        op::RenderMode::None,
        poseModel,
        false,
        0.6,
        0.7,
        0,
        model_dir,
        {op::HeatMapType::Parts},
        op::ScaleMode::ZeroToOne,
        false,
        0.05,
        -1,
          false,
          -1,
          "",
          "",
          0,
        false};
    opWrapper_.configure(wrapperStructPose);

    const op::WrapperStructFace wrapperStructFace{
      args_.compute_face(), op::Detector::Body, {368, 368}, op::RenderMode::None, 0.6, 0.7, 0.2};
    opWrapper_.configure(wrapperStructFace);

    const op::WrapperStructHand wrapperStructHand{args_.compute_hands(),
                                                  op::Detector::Body,
                                                  {368, 368},
                                                  args_.hand_num_scales(),
                                                  args_.hand_scale_gap(),
                                                  op::RenderMode::None,
                                                  0.6,
                                                  0.7,
                                                  0.2};
    opWrapper_.configure(wrapperStructHand);
    opWrapper_.start();

    result->set_success(true);
  }

  void execute(const scanner::BatchedElements& input_columns,
               scanner::BatchedElements& output_columns) override {
    auto& frame_col = input_columns[0];

    auto datumsPtr = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>();
    for (int i = 0; i < num_rows(frame_col); ++i) {
      datumsPtr->emplace_back();
      auto& datum = datumsPtr->at(datumsPtr->size() - 1);
      if (device_.type == DeviceType::GPU) {
        CUDA_PROTECT({
          cv::cuda::GpuMat gpu_input =
              scanner::frame_to_gpu_mat(frame_col[i].as_const_frame());
          datum->cvInputData = cv::Mat(gpu_input);
        });
      } else {
        datum->cvInputData =
            scanner::frame_to_mat(frame_col[i].as_const_frame());
      }
    }

    bool emplaced = opWrapper_.waitAndEmplace(datumsPtr);
    LOG_IF(FATAL, !emplaced) << "Failed to emplace pose work";
    std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumProcessed;
    bool popped = opWrapper_.waitAndPop(datumProcessed);
    LOG_IF(FATAL, !popped) << "Failed to pop pose results";

    for (auto& datum : *datumProcessed) {
      int num_people = datum->poseKeypoints.getSize(0);
      size_t size =
          num_people > 0 ? (POSE_SCORES + TOTAL_KEYPOINTS * 3) * num_people * sizeof(float) : 1 * sizeof(float);
      float* kp = new float[size / sizeof(float)];
      std::memset(kp, 0, size);
      float* curr_kp = kp;

      for (int i = 0; i < num_people; ++i) {
        std::memcpy(curr_kp,
                    datum->poseScores.getPtr() + i * POSE_SCORES,
                    POSE_SCORES * sizeof(float));
        curr_kp += POSE_SCORES;

        std::memcpy(curr_kp,
                    datum->poseKeypoints.getPtr() + i * POSE_KEYPOINTS * 3,
                    POSE_KEYPOINTS * 3 * sizeof(float));
        curr_kp += POSE_KEYPOINTS * 3;
        if (datum->faceKeypoints.getPtr() != nullptr) {
          std::memcpy(curr_kp,
                      datum->faceKeypoints.getPtr() + i * FACE_KEYPOINTS * 3,
                      FACE_KEYPOINTS * 3 * sizeof(float));
        }
        curr_kp += FACE_KEYPOINTS * 3;
        if (datum->handKeypoints[0].getPtr() != nullptr) {
          std::memcpy(curr_kp,
                      datum->handKeypoints[0].getPtr() + i * HAND_KEYPOINTS * 3,
                      HAND_KEYPOINTS * 3 * sizeof(float));
        }
        curr_kp += HAND_KEYPOINTS * 3;
        if (datum->handKeypoints[1].getPtr() != nullptr) {
          std::memcpy(curr_kp,
                      datum->handKeypoints[1].getPtr() + i * HAND_KEYPOINTS * 3,
                      HAND_KEYPOINTS * 3 * sizeof(float));
        }
        curr_kp += HAND_KEYPOINTS * 3;
      }

      float* device_kp = (float*)scanner::new_buffer(device_, size);
      scanner::memcpy_buffer((scanner::u8*)device_kp, device_, (scanner::u8*)kp,
                             scanner::CPU_DEVICE, size);
      scanner::insert_element(output_columns[0], (scanner::u8*)device_kp, size);
      delete kp;
    }
  }

 private:
  proto::OpenPoseArgs args_;
  scanner::DeviceHandle device_;
  op::Wrapper opWrapper_;
};

REGISTER_OP(OpenPose).frame_input("frame").output("pose", ColumnType::Bytes, "PoseList").protobuf_name(
    "OpenPoseArgs");

REGISTER_KERNEL(OpenPose, OpenPoseKernel)
    .device(scanner::DeviceType::CPU)
    .num_devices(BaseKernel::UnlimitedDevices)
    .batch();

REGISTER_KERNEL(OpenPose, OpenPoseKernel)
    .device(scanner::DeviceType::GPU)
    .num_devices(1)
    .batch();

}  // namespace scanner
