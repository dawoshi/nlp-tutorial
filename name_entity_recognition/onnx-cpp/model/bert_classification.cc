#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include "model/bert_classification.h"

namespace BertOnnx{

template <typename T>
int argmax(T begin, T end) {
    return std::distance(begin, std::max_element(begin, end));
}
BertClassification::BertClassification() {
  curr_sess_id_ = 0;
  session_inited_ = false;
}

BertClassification::~BertClassification() {

  std::cout << "~ BertClassification" << std::endl;
  if (session_inited_) session_inited_ = false;
  for(int i = 0; i < session_list_.size(); ++i) {
    delete session_list_[i];
  }
  session_list_.clear();
}

bool BertClassification::Init() {
  
  std::string model_file = "model.onnx";
  std::string vocab_file = "vocab.txt";
  int sess_thread_number = 8;
  int sess_number = 4;
  session_list_.clear();
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(sess_thread_number);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  for (int i = 0; i < sess_number; ++i) {
    Ort::Session* session =
      new Ort::Session(env, model_file.c_str(), session_options);
    session_list_.push_back(session);
  }
  session_inited_ = true;
  std::cout << "session init num thread: " << sess_thread_number << std::endl;
  tokenizer_.reset(new BertOnnx::FullTokenizer(vocab_file.c_str()));
  std::cout << "model init succuss!" << std::endl;
  return true;
}

std::vector<std::vector<int64_t>> BertClassification::build_input(const std::string& text) {
        std::vector<std::string> tokens;
        tokens.clear();
        tokenizer_->tokenize(text.c_str(), &tokens, 1000);
        std::vector<uint64_t> token_ids;
        token_ids.clear();
        tokenizer_->convert_tokens_to_ids(tokens, token_ids);
        std::vector<std::vector<int64_t>> res;
        std::vector<int64_t> input(32);
        std::vector<int64_t> mask(32);
        input[0] = 101;
        mask[0] = 1;
        for (int i = 0; i < token_ids.size() && i < 31; ++i) {
            input[i+1] = token_ids[i];
            mask[i+1] = token_ids[i] > 0;
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
    std::vector<int64_t> shape = {1, 32};
    auto& input_tensor_values = res[0];
    auto& mask_tensor_values = res[1];
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
  res = (idx >= 0 && idx < 10) ? BertOnnx::kNameTypes[idx] : "Unknown";
}
} // namespace
