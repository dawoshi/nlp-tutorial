#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>
#include "gflags.h"
#include "base/logging.h"
#include "text_classification/onnx-cpp/model/bert_classification.h"

using namespace nlp;

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  BertClassification detector;
  detector.Init();

  struct timeval t1, t2; 

  std::vector<std::string> courps;
  std::ifstream infile;
  infile.open("./test.txt");
  if (!infile.is_open())
    LOG(INFO) << "open file failure";
  while (!infile.eof()) {
    std::string line;
    while (getline(infile, line)) {
      courps.push_back(line);
    }   
  }   
  infile.close();
  LOG(INFO) << "test courps len count: " << courps.size();
  std::ofstream outfile;
  outfile.open("./out.txt");
  if(!outfile.is_open()) {
      return -1;
  }
  double total_time = 0.0;
  for (size_t i = 0; i < courps.size(); ++i) {
    outfile << courps[i];
    outfile << "\t";
    gettimeofday(&t1, NULL);
    std::string tmp;
    detector.predict(courps[i], tmp);
    gettimeofday(&t2, NULL);
    total_time += (t2.tv_sec - t1.tv_sec) +
      (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    for(int j = 0; j < tmp.size(); ++j) {
        outfile << tmp[j];
    }
    if(!tmp.empty()) {
        outfile << "\n";
    }
  }
  outfile.close();  
  LOG(INFO) << "Totle run Time : " << total_time * 1000.0 << "ms";
  return 0;
}
