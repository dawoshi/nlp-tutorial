#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: msra2mrc.py
import sys
sys.path.append("..")
import os
from utils.bmes_decode import bmes_decode
import json

def convert_file(input_file, output_file, tag2query_file):
    """
    Convert MSRA raw data to MRC format
    """
    origin_count = 0
    new_count = 0
    tag2query = json.load(open(tag2query_file, encoding='utf-8'))
    mrc_samples = []
    with open(input_file, encoding='utf-8') as fin:
        srcs = []  # 存放一个完整句子所有的token
        labels = []  # 存放token对应的label
        for line in fin:
            line = line.strip()
            if line:  # 当line非空时，即还没有遍历到一个句子的结束点，需要继续下一行获取该句子的token和label
                # origin_count += 1
                src, label = line.split(" ")
                srcs.append(src)
                labels.append(label)
            else:  # 当当前line为空，表明已获取完整的句子
                # 调用bmes_decode()函数进行解析，获取实体的开头和结尾的索引
                try:
                    tags = bmes_decode(char_label_list=[(char, label) for char, label in zip(srcs, labels)])
                    for i, (label, query) in enumerate(tag2query.items()):  # tag2query存放label与query的对应关系
                        start_position = [tag.begin for tag in tags if tag.tag == label]  # 获取当前lable实体的开始索引
                        end_position = [tag.end-1 for tag in tags if tag.tag == label]  # 获取当前lable实体的结束索引
                        span_position = [str(s)+';'+str(e) for s, e in zip(start_position, end_position)]  # 将开始索引和结束匹配
                        impossible = "true"
                        if start_position:
                            impossible = "false"
                        mrc_samples.append(
                            {
                                "qas_id": str(new_count) + "." + str(i+1),  # 数据id，小数点前面是问句的位数，小数点后面表示query的位数
                                "context": " ".join(srcs),  # 将srcs中所有的token用空格拼接，方便后续tokenizer.encoder_plus使用
                                "entity_label": label,  # 该标注数据中实体的label
                                "start_position": start_position,  # 所有实体的开始索引
                                "end_position": end_position,  # 所有实体的结束索引
                                "span_position": span_position,  # 所有实体的span
                                "impossible": impossible,  # 如果问句中存在label对应的实体就为fasle，否则没有实体就为true
                                "query": query  # 问句
                            }
                        )
                    new_count += 1
                    # 清空srcs和labels，用于存放下一个句子的token和label
                    srcs = []
                    labels = []
                except:
                   srcs = []
                   labels = []

    json.dump(mrc_samples, open(output_file, "w", encoding='utf-8'), ensure_ascii=False, sort_keys=True, indent=2)

def main():
    msra_raw_dir = "./mrc_data"
    msra_mrc_dir = "./mrc_data"
    tag2query_file = "queries/mrc.json"
    os.makedirs(msra_mrc_dir, exist_ok=True)
    for phase in ["train", "dev", "test"]:
        old_file = os.path.join(msra_raw_dir, f"{phase}.char.bmes")
        new_file = os.path.join(msra_mrc_dir, f"mrc-ner.{phase}")
        convert_file(old_file, new_file, tag2query_file)


if __name__ == '__main__':
    main()
