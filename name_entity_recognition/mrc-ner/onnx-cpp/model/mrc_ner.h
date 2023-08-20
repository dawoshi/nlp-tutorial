#ifndef MODEL_MRC_NER_H
#define MODEL_MRC_NER_H

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
	"NS", 
	"NR", 
	"NT",
  };
  const static std::vector<std::string> kQuery = {
	"按照地理位置划分的国家,城市,乡镇,大洲", 
	"人名和虚构的人物形象", 
	"组织包括公司,政府党派,学校,政府,新闻机构",
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

  class MrcNer {
    public:
        MrcNer();
        virtual ~MrcNer();
        bool Init();
        void predict(const std::string& text, std::vector<std::string> *res);
    private:
        void infer(const std::string &text,
			const std::string &query,
			const std::string &type,
			Ort::Session* session,
			std::vector<std::string>* res);
        void JoinStr(const std::vector<std::string> &tokens, 
			int start, int end, std::string &res);
        int curr_sess_id_;
        bool session_inited_;
	Ort::Env env;
	std::unordered_map<int, int> token_to_orig_map_;
        std::vector<Ort::Session*> session_list_;
        std::unique_ptr<base::FullTokenizer> tokenizer_ = nullptr;
        DISALLOW_COPY_AND_ASSIGN(MrcNer);
  };
} // namespace
#endif // MODEL_MRC_NER_H
