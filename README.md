# CVPR23 1st foundation model challenge Track2 4th
## Train

1. Using the RTMdet object detection model, the original training set and verification set are cleaned, the classification and discrimination of people and vehicles are completed, and the corresponding file names are renamed.
```bash
python train_det_cls.py
```

```bash
python val_det_cls.py
```

2. Generate the dataset into a json file and a jsonl file for Flickr30k.
Note: Please check the dataset path.
```bash
cd code
python data2flickr30k.py
```

3. Go through the steps above，we have splited the dataset into people and car json files and jsonl files.
The data format is as follows:
```
/dataset/train/
  train_images/            
    000001.jpg                
    ...  
    
  train_label.txt(Raw label)
  fliter_train_labels.txt(Cleaned and sorted labels)
  train_people.json
  train_car.json
  flickr30k_people.train.jsonl
  flickr30k_car.train.jsonl
  
  val_images/
    000002.jpg
    ...
    
  val_label.txt(Raw label)
  fliter_val_labels.txt(Cleaned and sorted labels)
  val_people.json
  val_car.json
  flickr30k_people.val.jsonl
  flickr30k_car.val.jsonl 
```

4. Start training.
The BEiT-3 **large** model can be finetuned on retrieval tasks using 2*3090:
```bash
cd code
sh train_car.sh
```

```bash
export CUDA_HOME=/usr/local/cuda-11.2
python -m torch.distributed.launch --nproc_per_node=2 run_beit3_finetuning.py \
        --model beit3_large_patch16_384 \
        --input_size 384 \
        --task flickr30k \
        --batch_size 80 \
        --layer_decay 0.85 \
        --lr 3e-5 \
        --epochs 17 \
        --warmup_epochs 3 \
        --drop_path 0.2 \
        --sentencepiece_model ./model_pth/beit3.spm \
        --finetune ./model_pth/beit3_large_patch16_384_f30k_retrieval.pth \
        --data_path ../dataset/train \
        --output_dir ./pt_out_car \
        --log_dir ./logs \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 1 \
        --enable_deepspeed \
        --checkpoint_activations \
        --category car
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`.
- `--finetune`: weight path of your pretrained models.
- `--task`: **flickr30k** for Flickr30k retrieval.
- `--output_dir`: The path to save the model and the log.txt file.
- `--epochs`: 15 for Flickr30k people retrieval.17 for Flickr30k car retrieval.
- `--warmup_epochs`: 5 for Flickr30k people retrieval, 3 for Flickr30k car retrieval.
- `--save_ckpt_freq`: How often the model is saved.（The first five rounds are not saved by default and can be set independently.）
- `--checkpoint_activations`: using gradient checkpointing for saving GPU memory.
- `--category`: Indicates the category (person or car) to train or infer about.

```bash
cd code
sh train_people.sh
```
```bash
export CUDA_HOME=/usr/local/cuda-11.2
python -m torch.distributed.launch --nproc_per_node=2 run_beit3_finetuning.py \
        --model beit3_large_patch16_384 \
        --input_size 384 \
        --task flickr30k \
        --batch_size 80 \
        --layer_decay 0.85 \
        --lr 3e-5 \
        --epochs 15 \
        --warmup_epochs 3 \
        --drop_path 0.2 \
        --sentencepiece_model ./model_pth/beit3.spm \
        --finetune ./model_pth/beit3_large_patch16_384_f30k_retrieval.pth \
        --data_path ../data/train \
        --output_dir ./pt_out_people \
        --log_dir ./logs \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 1 \
        --enable_deepspeed \
        --checkpoint_activations \
        --category people
```

## Infer
1.Generate json and jsonl files of the test dataset.
```bash
cd code
python jsonl_test.py
```
The data format is as follows:
```
/dataset/test/
  test_images/            
    000003.jpg                
    ...
       
  test_text.txt(test dataset raw txt file)
  test.txt(sorted txt files)
  test.json
  flickr30k.test.jsonl
```

2.Run the infer.sh file to infer the person and car respectively, and finally concat the two infer json files.
```bash
cd code
sh infer.sh
```

```bash
export CUDA_HOME=/usr/local/cuda-11.2
python infer.py \
        --model beit3_large_patch16_384 \
        --input_size 384 \
        --task flickr30k \
        --batch_size 96 \
        --sentencepiece_model ./model_pth/beit3.spm \
        --finetune_car /home/aistudio/data/data218751/car_lr3_s2_r75_16e.pt \
        --finetune_people /home/aistudio/data/data218751/mp_rank_00_model_states.pt \
        --data_path /home/aistudio/dataset/test/ \
        --eval \
        --dist_eval 
```
- `--sentencepiece_model`: the path of text tokenizer.
- `--finetune_car`: tha path of finetuning model of car.
- `--finetune_people`: the path of finetuning model of people.
- `--data_path`: the path of the test dataset.

The result files format is as follows:
```
/code/
    infer_json_car.json
    infer_json_people.json
    infer_json_all.json
```
The final infer file is infer_json_all.json.
