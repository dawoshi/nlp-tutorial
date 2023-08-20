#ifndef MODEL_SPAN_NER_H
#define MODEL_SPAN_NER_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "base/tokenization.h"
#include "onnxruntime_cxx_api.h"

namespace ner {
  const static std::vector<std::string> kNameTypes = {
	"unknown",
	"address", 
	"book", 
	"company",
	"game",
	"government",
	"movie",
	"name",
	"organization",
	"position",
	"scene"
  };
  // struct to represent the logit and offset of the answer related to context.
  struct Pos {
    Pos(int arg_start, int arg_end, float arg_logit)
        : start(arg_start), end(arg_end), logit(arg_logit) {}
    int start, end;
    float logit;
    bool operator<(const Pos& rhs) const { return rhs.logit < logit; }
  };
 // Returns the reversely sorted indices of a vector.
  template <typename T>
    std::vector<size_t> ReverseSortIndices(const T *v) {
      std::vector<size_t> idx;
      int i = 0;
      while(*v) {
	 idx.push_back(i++);
         ++v;
      }

      std::stable_sort(idx.begin(), idx.end(),
                   [&v](size_t i1, size_t i2) { return v[i2] < v[i1]; });
      return idx;
    }

  class SpanNer {
    public:
        SpanNer();
        virtual ~SpanNer();
        bool Init();
        void predict(const std::string& text, std::vector<std::string> *res);
    private:
        void infer(const std::string& text,
			std::vector<std::string>* res,
			Ort::Session* session);
        void JoinStr(const std::vector<std::string> &tokens, 
			int start, int end, std::string &res);
        int curr_sess_id_;
        bool session_inited_;
	Ort::Env env;
        std::vector<Ort::Session*> session_list_;
        std::unique_ptr<base::FullTokenizer> tokenizer_ = nullptr;
        DISALLOW_COPY_AND_ASSIGN(SpanNer);
  };
} // namespace
#endif // MODEL_SPAN_NER_H
