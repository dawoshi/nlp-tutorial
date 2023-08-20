## nlp-tutorial

<p align="center"><img width="100" src="https://upload.wikimedia.org/wikipedia/commons/c/c0/ONNX_logo_main.png" />  <img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

nlp-tutorial is a tutorial for who is studying NLP(Natural Language Processing)，This project uses Pytorch training and engineering deployment using C++.

## Current directory structure
-- base
-- data
-- third party
-- name entity recognition
-- text classification

### base
Base is pulled into many projects. For example, various ChromeOS daemons. So
the bar for adding stuff is that it must have demonstrated wide
applicability. Prefer to add things closer to where they're used (i.e. "not
base"), and pull into base only when needed.  In a project our size,
sometimes even duplication is OK and inevitable.

### name entity recognition
Named entity recognition includes span ner and mrc ner.
1、span ner is reference paper of SpanNER: Named EntityRe-/Recognition as Span Prediction[paper](https://arxiv.org/pdf/2106.00641.pdf), the code is reference of [https://github.com/lonePatient/BERT-NER-Pytorch], On the basis of this codes, I add the codes for converting to onnxruntime and deployment in C++.
2、Mrc ner is advances in Shannon.AI. for more details, please see A Unified MRC Framework for Named Entity Recognition In ACL 2020. [paper](https://arxiv.org/abs/1910.11476) , the code is in [https://github.com/ShannonAI/mrc-for-flat-nested-ner] , On the basis of this codes, I add the codes for converting to onnxruntime and deployment in C++.

### Text Classification
Using the pre trained models for text classification。
