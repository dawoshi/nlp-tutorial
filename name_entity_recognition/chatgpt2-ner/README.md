# 基于chatgpt进行命名实体的识别

## 训练数据下载

[msra_mrc](https://drive.google.com/file/d/1bAoSJfT1IBdpbQWSrZPjQPPbAsDGlN2D/view)

## 生成训练数据

- msra2chat.py
  
```
 {
   
   "prompt": "因有关日寇在京掠夺文物详情，藏界较为重视，也是我们收藏北京史料中的要件之一。上文中可能包括[NS,NR,NT]等类型的实体，如果有请提取出来。",
   "response": "实体类型是[NS]的词组有:[日]|[京]|[北京];",
   "history": []
 },
 {
   "prompt": "我们藏有一册１９４５年６月油印的《北京文物保存保管状态之调查报告》，调查范围涉及故宫、历博、古研所、北大清华图书馆、北图、日伪资料库等二十几家，言及文物二十万件以上，洋洋三万余言，是珍贵的北京史料。上文中可能包括[NS,NR,NT]等类型的实体，如果有请提取出来。",
   "response": "实体类型是[NS]的词组有:[北京]|[故宫]|[历博]|[北大清华图书馆]|[北图]|[日]|[北京];",
   "history": []
 },

```

## 模型训练

生成训练数据集后 具体的训练代码请查看[chatglm ptuning](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning)

## 预测

- predict.py
