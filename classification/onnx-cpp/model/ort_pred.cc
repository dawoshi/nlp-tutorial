#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include "onnxruntime_cxx_api.h"
#include "base/tokenization.h"

using namespace std;

const static std::vector<std::string> key = {
    "finance",
    "realty",
    "stocks",
    "education",
    "science",
    "society",
    "politics",
    "sports",
    "game",
    "entertainment"
};

template <typename T>
int argmax(const std::vector<T>& v) {
    if (v.empty()) {
        return -1;
    }
    return std::max_element(v.begin(), v.end()) - v.begin();
}
template <typename T>
int argmax(T a, T b) {
    return std::max_element(a, b) - a;
}


int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    const char* model_path = "model.onnx";
    const char* vocab_file = "./vocab.txt";
    std::string text = "飞鱼科技：推出保卫萝卜4等游戏，预期上半年除税后纯利3500万元至5000万元，同比扭亏";
    std::shared_ptr<BertOnnx::FullTokenizer> tokenizer_ = nullptr;
    tokenizer_.reset(new BertOnnx::FullTokenizer(vocab_file));
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
    Ort::Session session(env, model_path, session_options);
    size_t num_input_nodes = session.GetInputCount();
    std::cout<< num_input_nodes <<std::endl;
    std::cout<< session.GetOutputCount() <<std::endl;

    std::vector<int64_t> input_node_dims = {1, 32};

    size_t input_tensor_size = 32;
           
    // create input tensor object from data values ！！！！！！！！！！
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input.data(),
                                                            input_tensor_size, input_node_dims.data(), 2);

    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, mask.data(),
                                                            input_tensor_size, input_node_dims.data(), 2);

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
    ort_inputs.push_back(std::move(mask_tensor));

    std::vector<const char*> input_node_names = {"ids", "mask"};
    std::vector<const char*> output_node_names = {"output"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
                                    ort_inputs.size(), output_node_names.data(), 1);

    float* floatarr = output_tensors[0].GetTensorMutableData<float>();

    for (int i=0; i<10; i++)
    {
        std::cout<<floatarr[i]<<std::endl;
    }
    std::cout<< key[argmax(floatarr, floatarr+10)] << std::endl;

    return 0;
}
