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
#include "name_entity_recognition/span-ner/onnx-cpp/model/span_ner.h"


DEFINE_int32(model_max_len, 512, "pre_trained_model sequence length");
DEFINE_int32(sess_thread_number, 16, "session thread number");
DEFINE_int32(sess_number, 8, "session number");

DEFINE_string(model_file,
                "data/name_entity_recognition/span-ner/onnx-cpp/model/model.onnx",
                "onnx model file");
DEFINE_string(vocab_file,
                "data/name_entity_recognition/span-ner/onnx-cpp/model/vocab.txt",
                "pretrained model vocab file");


namespace ner {

const int label_size = 11;

template <typename T>
int argmax(T begin, T end) {
    return std::distance(begin, std::max_element(begin, end));
}
SpanNer::SpanNer() {
  curr_sess_id_ = 0;
  session_inited_ = false;
}

SpanNer::~SpanNer() {

  if (session_inited_) session_inited_ = false;
  for(int i = 0; i < session_list_.size(); ++i) {
    delete session_list_[i];
  }
  session_list_.clear();
}

bool SpanNer::Init() {
  
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

void SpanNer::infer(const std::string& text, std::vector<std::string>* res, 
  Ort::Session* session) {

  if(session == nullptr){
      return;
  }
  LOG(INFO) << text;
  std::vector<std::string> tokens;
  tokens.clear();
  tokenizer_->tokenize(text.c_str(), &tokens, 10000);
  tokens.insert(tokens.begin(), "[CLS]");
  tokens.push_back("[SEP]");
  std::vector<uint64_t> token_ids;
  token_ids.clear();
  tokenizer_->convert_tokens_to_ids(tokens, token_ids);

  size_t input_size = (int64_t)(token_ids.size());
  if(input_size > FLAGS_model_max_len) {
      input_size = FLAGS_model_max_len;
      token_ids[input_size-1] = 102;
  }
  std::vector<int64_t> shape = {1, input_size};

  std::vector<int64_t> input_tensor_values(input_size);
  std::vector<int64_t> mask_tensor_values(input_size);
  std::vector<int64_t> type_tensor_values(input_size);
  
  for (int i = 0; i < input_size; ++i) {
     input_tensor_values[i] = token_ids[i];
     mask_tensor_values[i] = (int64_t)1;
     type_tensor_values[i] = (int64_t)0;
  }


  const static auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
      	                                               OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, 
      	                                                input_tensor_values.data(),
                                                              input_tensor_values.size(),
      							shape.data(),
      							2);
  Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,
      	                                               mask_tensor_values.data(),
                                                             mask_tensor_values.size(),
      						       shape.data(),
      						       2);
  Ort::Value type_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,
      	                                               type_tensor_values.data(),
                                                             type_tensor_values.size(),
      						       shape.data(),
      						       2);
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_tensor));
  ort_inputs.push_back(std::move(mask_tensor));
  ort_inputs.push_back(std::move(type_tensor));
  
  const static std::vector<const char*> input_node_names =
    {"input_ids", "token_type_ids", "attention_mask"};
  const static std::vector<const char*> output_node_names = 
    {"start_logits","end_logits"};

  auto output_tensors = session->Run(Ort::RunOptions{nullptr},
    input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
    output_node_names.data(), 2);

  if (output_tensors.size() != output_node_names.size()) {
      return;
  }

  // logits = batch_size * seqlength * label_size

  const float* start_logits = output_tensors[0].GetTensorMutableData<float>();
  const float* end_logits = output_tensors[1].GetTensorMutableData<float>();

  std::vector<int> start_pred(tokens.size()), end_pred(tokens.size());

  // seqlen * label_size
  
  float start_logits_max, end_logits_max;
  int start_logits_idx, end_logits_idx;

  for(int i = 1; i< shape[1]-1; i++) {
    int new_token_idx = i * label_size;
    start_logits_max = start_logits[new_token_idx];
    end_logits_max = start_logits[new_token_idx];
    start_logits_idx = 0;
    end_logits_idx = 0;

    for (int j = 1; j < label_size; ++j) {
      int idx = new_token_idx + j;
      if (start_logits[idx] > start_logits_max){
        start_logits_max = start_logits[idx];
        start_logits_idx = j;
      }
      if(end_logits[idx] > end_logits_max) {
        end_logits_max = end_logits[idx];
        end_logits_idx = j;
      }
    }
    start_pred[i] = start_logits_idx;
    end_pred[i] = end_logits_idx;
  }
  std::string start_label, end_label;
  for (int i = 0; i < start_pred.size(); ++i) {
    start_label += std::to_string(start_pred[i]) + " ";
    end_label += std::to_string(end_pred[i]) + " ";
  }
  if (start_pred.size() != end_pred.size() &&
          start_pred.size() < 1) {
      return;
  }
  for (size_t i = 1; i < start_pred.size()-1; ++i) {
    if (start_pred[i] == 0) continue;
    for (size_t j = i+1; j < start_pred.size()-1; ++j) {
      if(end_pred[j] > 0 && start_pred[i] == end_pred[j]) {
	  std::string match;
	  JoinStr(tokens, i, j, match);
	  std::string type = ner::kNameTypes[start_pred[i]];
	  std::cout << text << "\t" 
	    	<< match << "\t" << type << std::endl;
	  std::string line = match + ":" + type;
	  res->push_back(line);
	  i = j;
	  break;
      }
    }
  }
}

void SpanNer::JoinStr(const std::vector<std::string> &tokens,
		int start, int end, std::string &res) {
	for(int i = start; i <= end && i <  tokens.size()-1; ++i) {
		res += tokens[i];
	}
}

void SpanNer::predict(const std::string& content, std::vector<std::string>* res) {
  if (!session_inited_) {
    return;
  }
  if (content.empty()) return;
  curr_sess_id_ = (curr_sess_id_ + 1) % session_list_.size();
  SpanNer::infer(content, res, session_list_[curr_sess_id_]);
}
} // namespace
