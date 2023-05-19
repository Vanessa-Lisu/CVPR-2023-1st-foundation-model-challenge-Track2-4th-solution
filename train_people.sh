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
        --data_path ../dataset/train \
        --output_dir ./pt_out_people \
        --log_dir ./logs \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 1 \
        --enable_deepspeed \
        --checkpoint_activations \
        --category people