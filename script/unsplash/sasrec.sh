DATASET="unsplash/without-query"
lr=1e-3

for decay in 0 1e-6 1e-4 1e-2
do
for dropout in 0 0.2 0.4 0.6 0.8
do
CUDA_VISIBLE_DEVICES=$1 python main.py \
    --dataset_code 'item_query' \
    --dataloader_code 'sasrec' \
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
    --lr ${lr} \
    --weight_decay ${decay} \
    --num_epochs 1 \
    --best_metric 'NDCG@10' \
    --model_init_seed 0 \
    --trm_dropout ${dropout}  \
    --trm_att_dropout ${dropout}  \
    --trm_user_dropout -1 \
    --trm_hidden_dim 64 \
    --trm_max_len 50 \
    --trm_num_blocks 2  \
    --trm_num_heads 2 \
    --verbose 10 \
    --experiment_dir "experiments/${DATASET}/dropout_${dropout}_decay_${decay}"
done
done