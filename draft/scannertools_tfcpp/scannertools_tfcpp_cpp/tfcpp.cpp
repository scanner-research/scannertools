#include "tensorflow/c/c_api.h"

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/types.pb.h"
#include "scanner/util/memory.h"
#include "scanner/util/serialize.h"

namespace scanner {

// https://stackoverflow.com/questions/41688217/how-to-load-a-graph-with-tensorflow-so-and-c-api-h-in-c-language
void free_buffer(void *data, size_t length) { free(data); }

TF_Buffer *read_file(const char *file) {
  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET); // same as rewind(f);

  void *data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer *buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;
  return buf;
}

void free_tensor(void *data, size_t len, void *arg) {}

#define TF_CHECK_RETURN(STATUS)                                                \
  {                                                                            \
    if (TF_GetCode(STATUS) != TF_OK) {                                         \
      result->set_success(false);                                              \
      std::string msg(TF_Message(STATUS));                                     \
      result->set_msg(msg);                                                    \
      return;                                                                  \
    }                                                                          \
  }

#define TF_CHECK_FAIL(STATUS)                                                  \
  {                                                                            \
    if (TF_GetCode(STATUS) != TF_OK) {                                         \
      LOG(FATAL) << TF_Message(STATUS);                                        \
    }                                                                          \
  }

class TFCPPKernel : public BatchedKernel {
public:
  TFCPPKernel(const KernelConfig &config)
      : BatchedKernel(config), device_(config.devices[0]) {}

  void setup_with_resources(proto::Result *result) override {
    std::string graph_path =
        "/tmp/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb";
    TF_Buffer *graph_def = read_file(graph_path.c_str());
    graph_ = TF_NewGraph();

    status_ = TF_NewStatus();
    TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph_, graph_def, opts, status_);
    TF_DeleteImportGraphDefOptions(opts);
    TF_CHECK_RETURN(status_);
    TF_DeleteBuffer(graph_def);

    TF_SessionOptions *sess_opts = TF_NewSessionOptions();

    session_ = TF_NewSession(graph_, sess_opts, status_);
    TF_CHECK_RETURN(status_);

    TF_DeleteSessionOptions(sess_opts);

    result->set_success(true);
  }

  ~TFCPPKernel() {
    TF_DeleteSession(session_, status_);
    TF_CHECK_FAIL(status_);
    TF_DeleteGraph(graph_);
    TF_DeleteStatus(status_);
  }

  TF_DataType frame_type_to_tf_type(FrameType type) {
    if (type == FrameType::U8) {
      return TF_UINT8;
    } else if (type == FrameType::F32) {
      return TF_FLOAT;
    } else {
      LOG(FATAL) << "Can't convert frame type " << type << " to TF_DataType";
    }
  }

  void execute(const BatchedElements &input_columns,
               BatchedElements &output_columns) override {

    TF_Output input_op = {TF_GraphOperationByName(graph_, "image_tensor"), 0};
    LOG_IF(FATAL, input_op.oper == nullptr) << "Failed to find input op";

    TF_Output output_op = {TF_GraphOperationByName(graph_, "detection_boxes"),
                           0};
    LOG_IF(FATAL, output_op.oper == nullptr) << "Failed to find output op";

    const Frame *frame = input_columns[0][0].as_const_frame();
    int64_t shape[FRAME_DIMS + 1];
    size_t batch_size = input_columns[0].size();
    shape[0] = batch_size;
    for (size_t i = 0; i < FRAME_DIMS; ++i) {
      shape[i + 1] = frame->shape[i];
    }

    u8* batch_buffer = new u8[frame->size() * batch_size];
    size_t frame_size = frame->size();
    for (size_t i = 0; i < batch_size; ++i) {
      std::memcpy(&batch_buffer[frame_size * i], input_columns[0][i].as_const_frame()->data,
                  frame_size);
    }

    TF_Tensor *input_tensor = TF_NewTensor(
        frame_type_to_tf_type(frame->type), shape, FRAME_DIMS + 1, batch_buffer,
        frame->size() * batch_size, free_tensor, nullptr);
    TF_Tensor *output_tensor = nullptr;

    TF_SessionRun(session_, nullptr, &input_op, &input_tensor, 1, &output_op,
                  &output_tensor, 1, nullptr, 0, nullptr, status_);
    TF_CHECK_FAIL(status_);

    delete batch_buffer;

    float *all_bboxes = (float *)TF_TensorData(output_tensor);
    int num_bboxes = TF_Dim(output_tensor, 1);
    int bbox_dim = TF_Dim(output_tensor, 2);
    for (size_t i = 0; i < batch_size; ++i) {
      std::vector<proto::BoundingBox> bboxes;
      float *frame_bboxes = &all_bboxes[i * num_bboxes * bbox_dim];
      for (size_t j = 0; j < num_bboxes; ++j) {
        proto::BoundingBox bbox;
        float *frame_bbox = &frame_bboxes[j * bbox_dim];
        bbox.set_x1(frame_bboxes[1]);
        bbox.set_y1(frame_bboxes[0]);
        bbox.set_x2(frame_bboxes[3]);
        bbox.set_y2(frame_bboxes[2]);
        bboxes.push_back(bbox);
      }

      u8 *buffer;
      size_t size;
      serialize_bbox_vector(bboxes, buffer, size);

      insert_element(output_columns[0], buffer, size);
    }
  }

private:
  DeviceHandle device_;
  TF_Graph *graph_;
  TF_Status *status_;
  TF_Session *session_;
};

REGISTER_OP(TFC).frame_input("frame").output("bytes", ColumnType::Bytes);

REGISTER_KERNEL(TFC, TFCPPKernel)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(BaseKernel::UnlimitedDevices);

REGISTER_KERNEL(TFC, TFCPPKernel)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);

} // namespace scanner
