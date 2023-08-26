# Onnxruntime deploy

## 介绍

1. 使用onnxruntime 对训练的模型进行c++部署

2. 使用bazel进行编译

## 环境

- bazel 2.0.0
- onnxruntime-linux-x64-1.13.1

## 效果

test courps len count: 10000

Totle run Time : 294441ms

title级别的推理速度达到 29ms

## 使用说明
```
# 编译：

bazel build //text_classification/onnx-cpp/model:model_test

# model.onnx 和 vocab.txt

位置：data/text_classification/onnx-cpp/model/

# 执行

nohup ./bazel-bin/text_classification/onnx-cpp/model/model_test

```

## 参考
[1] https://github.com/guodongxiaren/Bert-Chinese-Text-Classification-Pytorch
