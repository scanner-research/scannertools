#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/types.pb.h"
#include "scanner/util/memory.h"
#include "scanner/util/serialize.h"

#include <torch/torch.h>
#include <torch/script.h>

namespace scanner {

  float mu[] = {0.485, 0.456, 0.406};
  float sigma[] = {0.229, 0.224, 0.225};

class PyTorchKernel : public BatchedKernel {
public:
  PyTorchKernel(const KernelConfig &config)
      : BatchedKernel(config), device_(config.devices[0]) {}

  void setup_with_resources(proto::Result *result) override {
    module_ =
      torch::jit::load("/home/will/scannertools/scannertools_pytorch/script_module.pt");
    auto torchdevice = device_.type == DeviceType::GPU
      ? at::device({at::kCUDA, device_.id}) : at::device(at::kCPU);

    if (device_.type == DeviceType::GPU) {
      LOG_IF(FATAL, !torch::cuda::is_available())
        << "PyTorch kernel supposed to use GPU, but Torch CUDA API is not available";
      module_->to({at::kCUDA, device_.id});
    }

    mu_t_ = torch::from_blob(mu, {3}).unsqueeze(1).unsqueeze(2);
    sigma_t_ = torch::from_blob(sigma, {3}).unsqueeze(1).unsqueeze(2);
    if (device_.type == DeviceType::GPU) {
      mu_t_ = mu_t_.to(torchdevice);
      sigma_t_ = sigma_t_.to(torchdevice);
    }

    result->set_success(true);
  }

  void execute(const BatchedElements &input_columns,
               BatchedElements &output_columns) override {
    size_t batch_size = input_columns[0].size();
    const Frame* frame = input_columns[0][0].as_const_frame();
    LOG_IF(FATAL, frame->shape[0] != 224 || frame->shape[1] != 224)
      << "Frame must be 224 x 224, was " << frame->shape[0] << " x " << frame->shape[1];

    if (batch_buffer_ == nullptr) {
      batch_buffer_ = new_buffer(device_, batch_size * frame->size());
    }

    {
      ProfileBlock _block(profiler_, "make_batch");
      for (size_t i = 0; i < batch_size; ++i) {
        memcpy_buffer(&batch_buffer_[i * frame->size()], device_,
                      input_columns[0][i].as_const_frame()->data, device_,
                      frame->size());
      }
    }

    auto torchdevice = device_.type == DeviceType::GPU
      ? at::device({at::kCUDA, device_.id}) : at::device(at::kCPU);

    auto opts = torchdevice.dtype(at::kByte);
    at::Tensor input_tensor =
      torch::from_blob(
                       batch_buffer_,
                       {batch_size, frame->shape[0], frame->shape[1], frame->shape[2]},
                       opts);

    at::Tensor normalized_tensor;
    {
      ProfileBlock _block(profiler_, "preprocess");
      // convert to batch x channel x height x width and normalize to [0, 1]
      normalized_tensor = input_tensor.permute({0, 3, 1, 2}).to(at::kFloat) / 255.0;

      // normalize with mean/stddev. can't use torch::data::transforms::Normalize b/c it's only
      // implemented for CPU
      // https://caffe2.ai/doxygen-c/html/torch_2csrc_2api_2include_2torch_2data_2transforms_2tensor_8h_source.html
      normalized_tensor -= mu_t_;
      normalized_tensor /= sigma_t_;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(normalized_tensor);
    auto start = now();
    at::Tensor output = module_->forward(inputs).toTensor();
    profiler_->add_interval("forward", start, now());

    {
      ProfileBlock _block(profiler_, "copy_out");
      u8* output_buf = new_block_buffer(device_, output.nbytes(), batch_size);
      memcpy_buffer(output_buf, device_, (u8*) output.data_ptr(), device_, output.nbytes());

      size_t elem_size = output.nbytes() / batch_size;
      for (size_t i = 0; i < batch_size; ++i) {
        insert_element(output_columns[0], &output_buf[i * elem_size], elem_size);
      }
    }
  }

private:
  DeviceHandle device_;
  u8* batch_buffer_;
  std::shared_ptr<torch::jit::script::Module> module_;
  at::Tensor mu_t_;
  at::Tensor sigma_t_;
};

REGISTER_OP(PyTorch).frame_input("frame").output("bytes", ColumnType::Bytes);

REGISTER_KERNEL(PyTorch, PyTorchKernel)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(BaseKernel::UnlimitedDevices);

REGISTER_KERNEL(PyTorch, PyTorchKernel)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);

} // namespace scanner
