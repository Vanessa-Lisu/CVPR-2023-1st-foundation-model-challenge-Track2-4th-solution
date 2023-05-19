export CUDA_HOME=/usr/local/cuda-11.2

python -m torch.distributed.launch --nproc_per_node=1 infer.py \
        --model beit3_large_patch16_384 \
        --input_size 384 \
        --task flickr30k \
        --batch_size 128 \
        --sentencepiece_model ./model_pt/beit3.spm \
        --finetune_car ./pt_out_large/car-scale0.2-ratio0.75/checkpoint-16/mp_rank_00_model_states.pt \
        --finetune_people ./pt_out_large/car-scale0.2-ratio0.75/checkpoint-16/mp_rank_00_model_states.pt \
        --data_path /home/dell/zjx/ptp-main/ptp-main/dataset/test \
        --eval \
        --dist_eval
