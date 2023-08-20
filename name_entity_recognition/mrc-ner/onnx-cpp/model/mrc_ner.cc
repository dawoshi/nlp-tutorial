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
#include "name_entity_recognition/mrc-ner/onnx-cpp/model/mrc_ner.h"


DEFINE_int32(model_max_len, 512, "pre_trained_model sequence length");
DEFINE_int32(sess_thread_number, 16, "session thread number");
DEFINE_int32(sess_number, 8, "session number");

DEFINE_string(model_file,
		"data/name_entity_recognition/mrc-ner/onnx-cpp/model/model.onnx",
		"onnx model file");
DEFINE_string(vocab_file,
		"data/name_entity_recognition/mrc-ner/onnx-cpp/model/vocab.txt",
		"pretrained model vocab file");

namespace ner {

template <typename T>
int argmax(T begin, T end) {
    return std::distance(begin, std::max_element(begin, end));
}
MrcNer::MrcNer() {
  curr_sess_id_ = 0;
  session_inited_ = false;
}

MrcNer::~MrcNer() {
  if (session_inited_) session_inited_ = false;
  for(int i = 0; i < session_list_.size(); ++i) {
    delete session_list_[i];
  }
  session_list_.clear();
}

bool MrcNer::Init() {
  
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

void MrcNer::infer(const std::string &text, 
		const std::string &query,
		const std::string &cate,
		Ort::Session* session,
		std::vector<std::string>* res) {

    if(session == nullptr){
        return;
    }
    std::vector<std::string> tokens;
    std::vector<std::string> query_tokens;
    query_tokens.clear();
    tokenizer_->tokenize(query.c_str(), &query_tokens, 10000);
    query_tokens.insert(query_tokens.begin(), "[CLS]");
    query_tokens.push_back("[SEP]");
    tokens.insert(tokens.end(), query_tokens.begin(), query_tokens.end());
    const size_t query_len = query_tokens.size();
    query_tokens.clear();
    tokenizer_->tokenize(text.c_str(), &query_tokens, 10000);
    query_tokens.push_back("[SEP]");
    tokens.insert(tokens.end(), query_tokens.begin(), query_tokens.end());
    std::vector<uint64_t> token_ids;
    token_ids.clear();
    tokenizer_->convert_tokens_to_ids(tokens, token_ids); 
    
    size_t sz = tokens.size();
    if(sz > FLAGS_model_max_len) {
        sz = FLAGS_model_max_len;
	token_ids[sz-1] = 102;
    }
    std::vector<int64_t> input(sz);
    std::vector<int64_t> mask(sz);
    std::vector<int64_t> type(sz, (int64_t)0);
    for (int i = 0; i < sz; ++i) {
        input[i] = token_ids[i];
        mask[i] = token_ids[i] > 0;
    }
    std::vector<int64_t> input_node_dims = {1, sz};
    size_t input_tensor_size = sz;
    // create input tensor object from data values ！！！！！！！！！！
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input.data(),
                                                            input_tensor_size, input_node_dims.data(), 2);

    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, mask.data(),
                                                            input_tensor_size, input_node_dims.data(), 2);
    Ort::Value type_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, type.data(),
                                                            input_tensor_size, input_node_dims.data(), 2);

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
    ort_inputs.push_back(std::move(mask_tensor));
    ort_inputs.push_back(std::move(type_tensor));

    std::vector<const char*> input_node_names = {"input_ids", "token_type_ids", "attention_mask"};
    std::vector<const char*> output_node_names = {"start_logits","end_logits", "span_logits"};
    // curr_sess_id_ = (curr_sess_id_ + 1) % session_list_.size();
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
		    ort_inputs.size(), output_node_names.data(), 3);
    // auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
    //                                ort_inputs.size(), output_node_names.data(), 3);
    
    if (output_tensors.size() != output_node_names.size()) {
        return;
    }
    const float* start_logits = output_tensors[0].GetTensorMutableData<float>();
    const float* end_logits = output_tensors[1].GetTensorMutableData<float>();
    const float* span_logits = output_tensors[2].GetTensorMutableData<float>();
    auto span_shape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
    if(span_shape[1] != span_shape[2]){
        return;
    }
    std::string tmp = "";
    for(int i = query_len; i < span_shape[2]; ++i) {
        if(start_logits[i] > 0) {
            for(int j = i+1; j < span_shape[2]; ++j) {
                if(end_logits[j] > 0 && span_logits[i * span_shape[2] + j] > 0) {
                    std::string match;
                    JoinStr(tokens, i, j, match);
		    tmp += "{[" + std::to_string(i - query_len) +","+ std::to_string(j - query_len) +"],"+ match +"," + cate + "},";
                }
            }
        }
    }
    if(!tmp.empty()) {
        res->push_back(tmp);
    }
}

void MrcNer::JoinStr(const std::vector<std::string> &tokens,
    int start, int end, std::string &res) {
    for(int i = start; i <= end && i <  tokens.size()-1; ++i) {
        res += tokens[i];
    }
}

void MrcNer::predict(const std::string& content, std::vector<std::string>* res) {
  if (!session_inited_) {
    return;
  }
  if (content.empty()) return;
  std::string out = "[{content:" + content + "},";
  for(int i = 0; i < ner::kNameTypes.size(); ++i) {
     std::vector<std::string> out;
     curr_sess_id_ = (curr_sess_id_ + 1) % session_list_.size();
     MrcNer::infer(content, ner::kQuery[i] , ner::kNameTypes[i],
		     session_list_[curr_sess_id_], res);
     if(!out.empty()) {
        res->insert(res->end(), out.begin(), out.end());
     }
  }
  if(!res->empty()) {
      res->push_back("]");
      res->insert(res->begin(), out);
  }
}
} // namespace
