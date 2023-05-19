import re
import json
import os
import sys
from datasets import RetrievalDataset
from transformers import XLMRobertaTokenizer

file_path = "../data/val/val_people_labels.txt"
file_json_save_path = "../data/val/val_people.json"
split = 'val'
if os.path.exists(file_path):
    image_all = []
    with open(file_path) as f:
        sent_num = 0
        num = 0
        max = 5 #caption max len
        n = 0
        # image_all = []
        for line in f:
            line = line.strip()
            parts = line.split("$")
            image_name = parts[0]
            text =parts[2]
            # print(type(text))

            vert = re.split('[,.]', text)
            vert.pop()
            # print(vert)

            if len(vert) > max:
                print(vert) #输出跳过text超过五个的描述
                n+=1
                continue
            sentids = [i for i in range(sent_num, sent_num+len(vert))]

            sentences = []
            for i in range(len(vert)):
                words = []
                words.append(vert[i].lower().split(' '))
                new = [item for item in words[0] if item != '']
                # print(new)
                sentences_dict = {
                    'tokens' : new,
                    'raw' : vert[i],
                    'imgid' : num,
                    'sentid' : sentids[i]
                }
                sentences.append(sentences_dict)
                # print(sentences)

            image = {
                'sentids' : sentids,
                'imgid' : num,
                'sentences' : sentences,
                'split' : split,
                'filename' : image_name
            }
            num+=1
            sent_num += 1
            image_all.append(image)

    data = {
        'images' : image_all,
        'dataset' : 'flickr30k'
    }


    if os.path.exists(file_json_save_path):
        print("json file already exists.")
        os.remove(file_json_save_path)
        print("delect file success!")
    with open(file_json_save_path,'w') as f:
        json.dump(data,f)
        print("save file success!")

# tokenizer = XLMRobertaTokenizer("/home/has/lisu/beit3/model_pth/beit3.spm")
#
# RetrievalDataset.make_flickr30k_dataset_index(
#     data_path="../data/val",
#     tokenizer=tokenizer,
#     karpathy_path="../data/val",
# )


