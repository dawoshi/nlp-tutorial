#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>
#include "model/bert_classification.h"

using namespace std;
using namespace BertOnnx;

int main(){
  
  BertClassification detector;
  detector.Init();

  struct timeval t1, t2; 
  // gettimeofday(&t1, NULL);
  // std::string tmp;
  // std::string text = "财政部国家税务总局出台关于促进残疾人就业税收优惠政策的通知";
  // detector.predict(text, tmp);
  // std::cout << "predict: " << tmp << std::endl;
  // gettimeofday(&t2, NULL);
  // std::cout << "Totle run Time : " << ((t2.tv_sec - t1.tv_sec) +
  //   (double)(t2.tv_usec - t1.tv_usec)/1000000.0) *1000.0<< "ms" << std::endl;

  std::vector<string> courps;
  ifstream infile;
  infile.open("./people2014.txt");
  if (!infile.is_open())
    std::cout << "open file failure" << std::endl;
  while (!infile.eof()) {
    std::string line;
    while (getline(infile, line)) {
      courps.push_back(line);
    }   
  }   
  infile.close();
  std::cout << "test courps len count: " << courps.size() << std::endl;
  ofstream outfile;
  outfile.open("./clue_ner_span_test.txt");
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
    outfile << tmp;
    outfile << "\n";
  }   
  std::cout << "Totle run Time : " << total_time * 1000.0 << "ms" << std::endl;
  return 0;
}
