#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: msra2chat.py

# https://drive.google.com/file/d/1bAoSJfT1IBdpbQWSrZPjQPPbAsDGlN2D/view

import os
import json
import random
from collections import defaultdict

def encode_instruct(content, labels):

    desc = content + "上文中可能包括[%s]等类型的实体，如果有请提取出来。" % (labels)
    return desc

def encode_response(key_values):
    desc = ""
    for key, value in key_values.items():
        desc = "实体类型是[%s]的词组有:" % (key)
        for i, words in enumerate(value):
            desc = desc + "[%s]" % (words)
            if(i != len(value) -1):
                desc = desc + "|"
        desc = desc + ";"
    return desc

def convert_file(input_file, output_file):
    train_data = json.load(open(input_file))
    tree = lambda: defaultdict(tree)
    all_data = tree()

    for idx, sample in enumerate(train_data):
        span = sample["span_position"]
        label = sample["entity_label"]
        content = str(sample["context"]).replace(" ", "")
        values = []
        if(len(span)>1):
            for pair in span:
                start, end = pair.split(";")
                values.append(content[int(start):int(end)+1])
            all_data[content][str(label)] = values

    chat_samples = []
    labels = "NS,NR,NT"

    for key, value in all_data.items():
        chat_samples.append({
            "prompt": encode_instruct(key, labels),
            "response": encode_response(value),
            "history":[],
        })
    json.dump(chat_samples, open(output_file, "w", encoding='utf-8'), ensure_ascii=False, sort_keys=True, indent=2)
            
def main():
    msra_raw_dir = "./zh_msra"
    msra_chat_dir = "./chat_data"
    os.makedirs(msra_chat_dir, exist_ok=True)
    for phase in ["train", "dev", "test"]:
        old_file = os.path.join(msra_raw_dir, f"mrc-ner" + f".{phase}")
        new_file = os.path.join(msra_chat_dir, f"chat-ner.{phase}")
        convert_file(old_file, new_file)
        
if __name__ == '__main__':
    main()
