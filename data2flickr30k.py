import re
import json
import os
from datasets import RetrievalDataset
from transformers import XLMRobertaTokenizer

def create_json_file(file_path, file_json_save_path, split_name, category):
    if not os.path.exists(file_path):
        print("please check txt path!")
    else:
        image_all = []
        with open(file_path) as f:
            sent_num = 0
            num = 0
            n = 0
            # image_all = []
            for line in f:
                line = line.strip()
                parts = line.split("$")
                image_name = parts[0]
                if category not in image_name:
                    continue
                else:
                    text = parts[2]
                    sentids = [sent_num]
                    words = re.split('[,. ]', text.lower())
                    new = [item for item in words if item != '']
                    # print(new)
                    sentences = []
                    sentences_dict = {
                        'tokens': new,
                        'raw': text,
                        'imgid': num,
                        'sentid': sent_num

                    }
                    sentences.append(sentences_dict)
                    image = {
                        'sentids': sentids,
                        'imgid': num,
                        'sentences': sentences,
                        'split': split_name,
                        'filename': image_name
                    }
                    num += 1
                    sent_num += 1
                    image_all.append(image)
                    # print(image)
        print(len(image_all))
        data = {
            'images': image_all,
            'dataset': 'flickr30k'
        }
        if os.path.exists(file_json_save_path):
            print("{} json file already exists.".format(split_name))
            os.remove(file_json_save_path)
            print("delect file success!")
        with open(file_json_save_path,'w') as f:
            json.dump(data,f)
            print("save {} json file success!".format(split_name))

def create_jsonl_file(data_path, json_path, category):

    tokenizer = XLMRobertaTokenizer("/home/has/lisu/beit3/model_pth/beit3.spm")

    RetrievalDataset.make_flickr30k_dataset_index(
        data_path=data_path,
        tokenizer=tokenizer,
        json_path=json_path,
        category=category
    )


if __name__ == '__main__':
    # train.caption.txt文件路径
    train_txt_exsist_path = "../data/train/filtered_labels.txt"

    # train.json存储地址
    train_people_json_save_path = "../data/train/train_people.json"
    train_car_json_save_path = "../data/train/train_car.json"

    # val.caption.txt文件路径
    val_txt_exsist_path = "../data/val/filtered_labels.txt"

    # val.json存储地址
    val_people_json_save_path = "../data/val/val_people.json"
    val_car_json_save_path = "../data/val/val_car.json"

    #train、val生成json文件
    create_json_file(train_txt_exsist_path, train_people_json_save_path, 'train', '0')
    # create_json_file(train_txt_exsist_path, train_car_json_save_path, 'train', 'v')
    create_json_file(val_txt_exsist_path, val_people_json_save_path, 'val', '0')
    # create_json_file(val_txt_exsist_path, val_car_json_save_path, 'val', 'v')

    #train、val生成jsonl文件
    train_image_path = "../data/train"
    val_image_path = "../data/val"
    create_jsonl_file(train_image_path, train_people_json_save_path, '0')
    # create_jsonl_file(train_image_path, train_car_json_save_path, '0')
    create_jsonl_file(val_image_path, val_people_json_save_path, '0')
    # create_jsonl_file(val_image_path, val_car_json_save_path, '0')
