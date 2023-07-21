#ifndef MODEL_MODEL_H
#define MODEL_MODEL_H
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include "model/common.h"
#include "base/tokenization.h"
#include "onnxruntime_cxx_api.h"

namespace BertOnnx {
  const static std::vector<std::string> kNameTypes = {
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
  class BertClassification {
    public:
        BertClassification();
        virtual ~BertClassification();
        bool Init();
        void predict(const std::string& text, std::string &res);
    private:
        int infer(const std::string& text, float* score, Ort::Session* session);
        std::vector<std::vector<int64_t>> build_input(const std::string& text);
        int curr_sess_id_;
        bool session_inited_;
        std::vector<Ort::Session*> session_list_;
        std::shared_ptr<BertOnnx::FullTokenizer> tokenizer_ = nullptr;
        // DISALLOW_COPY_AND_ASSIGN(BertClassification);
  };
} // namespace
#endif // MODEL_MODEL_H
