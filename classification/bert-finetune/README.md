# Bert-Chinese-Text-Classification-Pytorch

## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX  


## 中文数据集
我从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。数据以字为单位输入模型。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万

## 效果

模型|acc|备注
--|--|--
bert|94.83%|单纯的bert
ERNIE|94.61%|说好的中文碾压bert呢  
bert_CNN|94.44%|bert + CNN  
bert_RNN|94.57%|bert + RNN  
bert_RCNN|94.51%|bert + RCNN  
bert_DPCNN|94.47%|bert + DPCNN  

原始的bert效果就很好了，把bert当作embedding层送入其它模型，效果反而降了，之后会尝试长文本的效果对比。

## 预训练语言模型
bert模型放在 bert_pretain目录下，ERNIE模型放在ERNIE_pretrain目录下，每个目录下都是三个文件：
 - pytorch_model.bin  
 - bert_config.json  
 - vocab.txt  

预训练模型下载地址：  
bert_Chinese: 模型 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz  
              词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt  
来自[这里](https://github.com/huggingface/pytorch-transformers)   
备用：模型的网盘地址：https://pan.baidu.com/s/1qSAD5gwClq7xlgzl_4W3Pw

ERNIE_Chinese: http://image.nghuyong.top/ERNIE.zip  
来自[这里](https://github.com/nghuyong/ERNIE-Pytorch)  
备用：网盘地址：https://pan.baidu.com/s/1lEPdDN1-YQJmKEd_g9rLgw  

解压后，按照上面说的放在对应目录下，文件名称确认无误即可。  

## 使用说明
下载好预训练模型就可以跑了。
```
# 训练并测试：
# bert
python run.py --model bert

# bert + 其它
python run.py --model bert_CNN

# ERNIE
python run.py --model ERNIE
```

### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。  

## 对应论文
[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
[2] ERNIE: Enhanced Representation through Knowledge Integration  

## 参考
[1] https://github.com/guodongxiaren/Bert-Chinese-Text-Classification-Pytorch
