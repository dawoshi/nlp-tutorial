#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include "base/logging.h"
#include "base/files/file_path.h"
#include "base/files/file_util.h"
#include "model/mrc_bert_ner.h"


namespace ner {

template <typename T>
int argmax(T begin, T end) {
    return std::distance(begin, std::max_element(begin, end));
}
MrcBertNer::MrcBertNer() {
  session_inited_ = false;
}

MrcBertNer::~MrcBertNer() {}

bool MrcBertNer::Init() {
  
  std::string model_file = "model.onnx";
  std::string vocab_file = "vocab.txt";
  if(!base::PathExists(base::FilePath(model_file)) 
		  || !base::PathExists(base::FilePath(vocab_file))){
      LOG(ERROR) << "model file or vocab file not exist please check it";
      return false;
  }
  int sess_thread_number = 8;
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(sess_thread_number);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_ = std::make_unique<Ort::Session>(env_, model_file.c_str(), session_options);
  session_inited_ = true;
  LOG(INFO) << "session init num thread: " << sess_thread_number;
  tokenizer_.reset(new BertOnnx::FullTokenizer(vocab_file.c_str()));
  LOG(INFO) << "model init succuss!";
  return true;
}

void MrcBertNer::infer(const std::string &text, 
		const std::string &query,
		const std::string &type,
		std::vector<std::string>* res) {
    std::vector<std::string> tokens;
    std::vector<std::string> query_tokens;
    tokenizer_->tokenize(query.c_str(), &query_tokens, 2048);
    query_tokens.insert(query_tokens.begin(), "[CLS]");
    query_tokens.push_back("[SEP]");
    tokens.insert(tokens.end(), query_tokens.begin(), query_tokens.end());
    std::vector<std::string> text_tokens;
    tokenizer_->tokenize(text.c_str(), &text_tokens, 2048);
    text_tokens.push_back("[SEP]");
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    std::vector<uint64_t> tokens_ids;
    tokenizer_->convert_tokens_to_ids(tokens, tokens_ids);


    std::vector<int64_t> input_tensor_values;
    std::vector<int64_t> mask_tensor_values;
    std::vector<int64_t> type_tensor_values;
    
    for (int i = 0; i < tokens_ids.size(); ++i) {
       input_tensor_values.push_back((int64_t)tokens_ids[i]);
       mask_tensor_values.push_back((int64_t)1);
       type_tensor_values.push_back((int64_t)0);
    }
    LOG(INFO) << "token length: " << tokens.size() << "\t" << "token id length: " << input_tensor_values.size();
    LOG(INFO) << "Tokens: ";
    for(int i =0; i < tokens.size(); ++i) {
        std::cout << tokens[i] << " ";
    }
    std::cout << std::endl;
    LOG(INFO) << "token Id: ";
    for(int i =0; i < input_tensor_values.size(); ++i) {
	    std::cout << input_tensor_values[i] << " ";
    }
    std::cout << std::endl;

    const size_t input_size = (int64_t)(input_tensor_values.size());
    std::vector<int64_t> shape = {1, input_size};

    const static auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                   OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, 
                                                                input_tensor_values.data(),
                                                                input_tensor_values.size(),
                                                                shape.data(),
                                                                shape.size());
    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,
                                                               mask_tensor_values.data(),
                                                               mask_tensor_values.size(),
                                                               shape.data(),
                                                               shape.size());
    Ort::Value type_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,
                                                               type_tensor_values.data(),
                                                               type_tensor_values.size(),
                                                               shape.data(),
                                                               shape.size());
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
    ort_inputs.push_back(std::move(mask_tensor));
    ort_inputs.push_back(std::move(type_tensor));
    
    const static std::vector<const char*> input_node_names =
      {"input_ids", "token_type_ids", "attention_mask"};
    const static std::vector<const char*> output_node_names = 
      {"start_logits","end_logits", "span_logits"};
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
      input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
      output_node_names.data(), output_node_names.size());
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
    for(int i = query_tokens.size(); i < span_shape[2]; ++i) {
        if(start_logits[i] > 0) {
            for(int j = i+1; j < span_shape[2]; ++j) {
                if(end_logits[j] > 0 && span_logits[i * span_shape[2] + j] > 0) {
                    std::string match;
                    JoinStr(tokens, i, j, match); 
                    // std::cout << text << "\t"
                    //    << match << "\t" << type << std::endl;
                    std::string line = text + "\t" + match + "\t" + type;
                    res->push_back(line);
                }
            }
        }
    }
}

void MrcBertNer::JoinStr(const std::vector<std::string> &tokens,
    int start, int end, std::string &res) {
    for(int i = start; i <= end && i <  tokens.size()-1; ++i) {
        res += tokens[i];
    }
}

void MrcBertNer::predict(const std::string& content, std::vector<std::string>* res) {
  if (!session_inited_) {
    return;
  }
  if (content.empty()) return;
  for(int i = 0; i < ner::kNameTypes.size(); ++i) {
     std::vector<std::string> out;
     MrcBertNer::infer(content, ner::kQuery[i] , ner::kNameTypes[i], res);
     res->insert(res->end(), out.begin(), out.end());
  }
}
} // namespace
