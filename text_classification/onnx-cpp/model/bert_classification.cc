#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include "gflags.h"
#include "base/logging.h"
#include "base/files/file_path.h"
#include "base/files/file_util.h"
#include "text_classification/onnx-cpp/model/bert_classification.h"


DEFINE_int32(model_max_len, 512, "pre_trained_model sequence length");
DEFINE_int32(sess_thread_number, 16, "session thread number");
DEFINE_int32(sess_number, 8, "session number");

DEFINE_string(model_file,
		"data/text_classification/onnx-cpp/model/model.onnx",
		"onnx model file");
DEFINE_string(vocab_file,
		"data/text_classification/onnx-cpp/model/vocab.txt",
		"pretrained model vocab file");

namespace nlp{

template <typename T>
int argmax(T begin, T end) {
    return std::distance(begin, std::max_element(begin, end));
}
BertClassification::BertClassification() {
  curr_sess_id_ = 0;
  session_inited_ = false;
}

BertClassification::~BertClassification() {
  if (session_inited_) session_inited_ = false;
  for(int i = 0; i < session_list_.size(); ++i) {
    delete session_list_[i];
  }
  session_list_.clear();
}

bool BertClassification::Init() {
  
  std::string model_file = FLAGS_model_file;
  std::string vocab_file = FLAGS_vocab_file;
  if(!base::PathExists(base::FilePath(model_file)) 
        	  || !base::PathExists(base::FilePath(vocab_file))){
          LOG(INFO) << "model file or vocab file not exist please check it";
      return false;
  }

  session_list_.clear();
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(FLAGS_sess_thread_number);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  for (int i = 0; i < FLAGS_sess_number; ++i) {
    auto session =
      new Ort::Session(env, model_file.c_str(), session_options);
    session_list_.push_back(session);
  }
  session_inited_ = true;
  LOG(INFO) << "session init num thread: " << FLAGS_sess_thread_number;
  tokenizer_.reset(new base::FullTokenizer(vocab_file.c_str()));
  LOG(INFO) << "model init succuss!";
  return true;
}

std::vector<std::vector<int64_t>> BertClassification::build_input(const std::string& text) {
        std::vector<std::string> tokens;
        tokens.clear();
        tokenizer_->tokenize(text.c_str(), &tokens, 10000);
	tokens.push_back("[SEP]");
	tokens.insert(tokens.begin(), "[CLS]");
        std::vector<uint64_t> token_ids;
        token_ids.clear();
        tokenizer_->convert_tokens_to_ids(tokens, token_ids);
        std::vector<std::vector<int64_t>> res;
        size_t sz = tokens.size();
	if(sz > FLAGS_model_max_len) {
	    sz = FLAGS_model_max_len;
	    token_ids[sz-1] = 102;
	}
        std::vector<int64_t> input(sz);
        std::vector<int64_t> mask(sz);
        for (int i = 0; i < sz; ++i) {
            input[i] = token_ids[i];
            mask[i] = token_ids[i] > 0;
        }
        res.push_back(std::move(input));
        res.push_back(std::move(mask));
        return res;
}

int BertClassification::infer(const std::string& text, float* score, Ort::Session* session) {

    if(session == nullptr){
        return -1;
    }
    // 调用前面的build_input
    auto res = build_input(text);
    auto& input_tensor_values = res[0];
    auto& mask_tensor_values = res[1];
    const size_t sz = input_tensor_values.size();
    std::vector<int64_t> shape = {1, sz};
    const static auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_tensor_values.data(),
                                                            input_tensor_values.size(), shape.data(), 2);
    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, mask_tensor_values.data(),
                                                            mask_tensor_values.size(), shape.data(), 2);
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
    ort_inputs.push_back(std::move(mask_tensor));
    const static std::vector<const char*> input_node_names = {"ids", "mask"};
    const static std::vector<const char*> output_node_names = {"output"};
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
                                    ort_inputs.size(), output_node_names.data(), 1);
    if (output_tensors.size() != output_node_names.size()) {
        return -1;
    }
    const float* output = output_tensors[0].GetTensorData<float>();
    int idx = argmax(output, output+10);
    if (score != nullptr) {
        *score = output[idx];
    }
    return idx;
}

void BertClassification::predict(const std::string& content, std::string &res) {
  if (!session_inited_) {
    return;
  }
  if (content.empty()) return;
  curr_sess_id_ = (curr_sess_id_ + 1) % session_list_.size();
  float score;
  int idx = BertClassification::infer(content, &score, session_list_[curr_sess_id_]);
  res = (idx >= 0 && idx < 10) ? nlp::kNameTypes[idx] : "Unknown";
}
} // namespace
