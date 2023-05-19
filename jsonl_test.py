import os
import re
import json
from datasets import RetrievalDataset
from transformers import XLMRobertaTokenizer


def orderTxt(txt_path, save_path):
    # save_path = '/home/dell/zjx/ptp-main/ptp-main/dataset/test/ordered_test_text.txt'
    car = []
    people = []
    num_car = 0
    with open(txt_path) as f:
        for line in f:
            if 'He ' in line or 'She ':
                people.append(line)
            else:
                car.append(line)
                num_car += 1

    if os.path.exists(save_path):
        os.remove(save_path)
    else:
        print("The file does not exist")

    with open(save_path, 'a') as f:
        for line in car:
            f.write(line)
        for line in people:
            f.write(line)

    # return num_car

def create_test_json(path_imgs, texts_root, file_json_save_path):
    # path_imgs = '../data/test/test_images'
    # texts_root = "../data/test/test_text.txt"
    # bulid json dict
    imgs_name = []
    # for files in os.listdir(path_imgs):
    #     imgs_name.append(files)
    #     # print(files)
    files = os.listdir(path_imgs)
    files.sort()
    for file_name in files:
        imgs_name.append(file_name)

    captions = []
    f = open(texts_root, 'r')
    for line in f.readlines():
        text = line.strip()
        captions.append(text)
    print(len(imgs_name))
    print(len(captions))
    # print(captions)

    image_all = []
    split = 'test'
    for i in range(len(captions)):
        words = re.split('[,. ]', captions[i].lower())
        new_words = [item for item in words if item != '']
        # print(new_words)
        sentences = []
        sentences_dict = {
            'tokens': new_words,
            'raw': captions[i],
            'imgid': i,
            'sentid': i
        }
        sentids = [i]
        sentences.append(sentences_dict)

        image = {
            'sentids': sentids,
            'imgid': i,
            'sentences': sentences,
            'split': split,
            'filename': imgs_name[i]
        }
        image_all.append(image)

    data = {
        'images': image_all,
        'dataset': 'flickr30k'
    }
    # save json file
    # file_json_save_path = "../data/test/test.json"
    if os.path.exists(file_json_save_path):
        print("test json file already exists.")
        os.remove(file_json_save_path)
        print("delect file success!")
    with open(file_json_save_path, 'w') as f:
        json.dump(data, f)
        print("save test json file success!")

if __name__ == '__main__':
    #test图片路径
    path_imgs = '../data/test/test_images'
    #test文本描述文件路径
    texts_root = "../data/test/test_text.txt"
    # 对原始的描述文件重排序
    save_txt_root = "../data/test/test.txt"
    orderTxt(texts_root, save_txt_root)
    #test json文件保存路径
    file_json_save_path = "../data/test/test.json"

    create_test_json(path_imgs,  save_txt_root, file_json_save_path)

    #build test jsonl file
    tokenizer = XLMRobertaTokenizer("./model_pth/beit3.spm")

    RetrievalDataset.make_flickr30k_dataset_index(
        data_path="../data/test",#test路径
        tokenizer=tokenizer,
        json_path=file_json_save_path,
        category=None
    )