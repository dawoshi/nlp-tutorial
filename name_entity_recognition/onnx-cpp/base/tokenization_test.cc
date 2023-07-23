#include<iostream>
#include <memory>
#include "tokenization.h"
using namespace std;
int main(){
    const char* vocab_file = "./vocab.txt";
    // auto tokenizer = new BertOnnx::FullTokenizer(vocab_file.c_str());
    std::shared_ptr<BertOnnx::FullTokenizer> tokenizer_ = nullptr;
    tokenizer_.reset(new BertOnnx::FullTokenizer(vocab_file));
    std::vector<std::string> tokens;
    std::string text = "李稻葵:过去2年抗疫为每人增寿10天";
    tokenizer_->tokenize(text.c_str(), &tokens, 100);
    vector<uint64_t> ids;
    tokenizer_->convert_tokens_to_ids(tokens, ids);
    for (size_t i = 0; i < ids.size(); i++) {
        if (i != 0) std::cout << " ";
        std::cout << ids[i];
    }
    std::cout<<std::endl; 
    return 0;
}
