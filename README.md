## nlp-tutorial

<p align="center">
  <img width="100" src="https://upload.wikimedia.org/wikipedia/commons/c/c0/ONNX_logo_main.png" />  
  <img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" />
  <img width="100" src="https://upload.wikimedia.org/wikipedia/en/7/7d/Bazel_logo.svg" />
  <img width="100" src = "https://upload.wikimedia.org/wikipedia/commons/1/18/ISO_C%2B%2B_Logo.svg" />
</p>

nlp-tutorial is a tutorial for who is studying NLP(Natural Language Processing)，This project uses Pytorch training and engineering deployment using C++.

## Structures
- base
- third party
- name entity recognition
- text classification
- data

```text
├── base
├── third party
├── name entity recognition
|  |  └── span ner
|  |   |   └── train-pytorch(span-bert-ner-pytorch)
|  |   |   └── onnx-cpp 
|  |  └── mrc ner
|  |   |   └── train-pytorch(mrc-for-flat-nested-ner)
|  |   |   └── onnx-cpp 
├── text classification
|  |  └── bert classification
|  |   |   └── train-pytorch
|  |   |   └── onnx-cpp 
├── data
|  |  └── ......
```

### base
Base is pulled into many projects. For example, various ChromeOS daemons. So
the bar for adding stuff is that it must have demonstrated wide
applicability. Prefer to add things closer to where they're used (i.e. "not
base"), and pull into base only when needed.  In a project our size,
sometimes even duplication is OK and inevitable.

### name entity recognition
Named entity recognition includes span ner and mrc ner.

1、span ner is reference paper of SpanNER: Named EntityRe-/Recognition as Span Prediction [paper](https://arxiv.org/pdf/2106.00641.pdf), the code is reference of [https://github.com/lonePatient/BERT-NER-Pytorch], On the basis of this codes, I add the codes for converting to onnxruntime and deployment in C++.


#### CLUENER

The overall performance of BERT on **dev**:

|              | Accuracy (entity)  | Recall (entity)    | F1 score (entity)  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.7897     | 0.8031     | 0.7963    |
| BERT+CRF     | 0.7977 | 0.8177 | 0.8076 |
| BERT+Span    | 0.8132 | 0.8092 | 0.8112 |
| BERT+Span+adv    | 0.8267 | 0.8073 | **0.8169** |
| BERT-small(6 layers)+Span+kd    | 0.8241 | 0.7839 | 0.8051 |
| BERT+Span+focal_loss    | 0.8121 | 0.8008 | 0.8064 |
| BERT+Span+label_smoothing   | 0.8235 | 0.7946 | 0.8088 |

2、Mrc ner is advances in Shannon.AI. for more details, please see A Unified MRC Framework for Named Entity Recognition In ACL 2020. [paper](https://arxiv.org/abs/1910.11476) , the code is in [https://github.com/ShannonAI/mrc-for-flat-nested-ner] , On the basis of this codes, I add the codes for converting to onnxruntime and deployment in C++.

#### msra_zh

|   model  | precision  | Recall | F1 score  |
| -------- | ---------- | ------ | --------- |
| BERT+MRC | 0.9243     | 0.9113 | 0.9177    |



### Text Classification

Using the pre trained models for text classification。

#### THUCNews

model|acc|remarks
--|--|--
bert|94.83%|单纯的bert
ERNIE|94.61%|说好的中文碾压bert呢  
bert_CNN|94.44%|bert + CNN  
bert_RNN|94.57%|bert + RNN  
bert_RCNN|94.51%|bert + RCNN  
bert_DPCNN|94.47%|bert + DPCNN  
