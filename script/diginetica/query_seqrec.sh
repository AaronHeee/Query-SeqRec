DATASET="diginetica/with-query"

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --dataset_code 'item_query' \
    --dataloader_code 'aug_sasrec' \
    --trainer_code "sasrec_all" \
    --model_code 'sasrec' \
    --data_path "data/${DATASET}" \
    --train_batch_size 128 \
    --val_batch_size 128 \
    --test_batch_size 128 \
    --train_negative_sampler_code 'random' \
    --train_negative_sample_size 0 \
    --train_negative_sampling_seed 0 \
    --test_negative_sampler_code 'random' \
    --test_negative_sample_size 100 \
    --test_negative_sampling_seed 98765 \
    --device 'cuda' \
    --device_idx $1 \
    --optimizer 'Adam' \
    --lr 1e-3 \
    --weight_decay 1e-6 \
    --num_epochs 1000 \
    --best_metric 'NDCG@10' \
    --model_init_seed 0 \
    --trm_dropout 0.6 \
    --trm_att_dropout 0.6 \
    --trm_user_dropout 0 \
    --item_label_replace_prob 0 \
    --item_input_replace_prob 0.25 \
    --query_input_replace_prob 0.25 \
    --user_input_replace_prob 0 \
    --trm_hidden_dim 64 \
    --trm_max_len 50 \
    --trm_num_blocks 2  \
    --trm_num_heads 2 \
    --graph_based_prob 0.2 \
    --verbose 10 \
    --experiment_dir "experiments/${DATASET}/"
