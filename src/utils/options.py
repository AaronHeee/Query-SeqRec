from src.datasets import DATASETS
from src.dataloaders import DATALOADERS
from src.models import MODELS
from src.trainers import TRAINERS

import argparse

parser = argparse.ArgumentParser(description='Query-SeqRec')

################
# Test
################
parser.add_argument('--load_pretrained_weights', type=str, default=None)
parser.add_argument('--eval_mode', type=str, default=None)

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='item_query', choices=DATASETS.keys())
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')
parser.add_argument('--dataset_split_seed', type=int, default=98765)
parser.add_argument('--data_path', type=str, default='data/unsplash_lite')

################
# Dataloader
################
parser.add_argument('--dataloader_code', type=str, default='sasrec', choices=DATALOADERS.keys())
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)

################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['random', 'vse'],
                    help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=0)
parser.add_argument('--train_negative_sampling_seed', type=int, default=0)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['random', 'vse'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=98765)

################
# Trainer
################
parser.add_argument('--trainer_code', type=str, default='sasrec_sample', choices=TRAINERS.keys())
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
parser.add_argument('--emb_device_idx', type=str, default=None, 
                    help="None: as the same as device_idx; cpu: move all to the cpu mem; \
                            {'cpu':(0,16), 'cuda:0':(16,64)}: Embed[:16] on cpu and Embed[16:64] on cuda:0")

# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD','Adam', 'Adagrad'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Adam Epsilon')

# training #
parser.add_argument('--verbose', type=int, default=10)
# training on large gpu #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
# training on small gpu #
parser.add_argument('--global_epochs', type=int, default=1000, help='Number of epochs for global training')
parser.add_argument('--local_epochs', type=int, default=10, help='Number of epochs for local training')
parser.add_argument('--subset_size', type=int, default=1000, help='Maximal Items Size')

# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')

################
# Model
################
parser.add_argument('--model_code', type=str, default='sasrec', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=int, default=0)
parser.add_argument('--early_query', type=int, default=0)
# Transformer Blocks #
parser.add_argument('--trm_max_len', type=int, default=200, help='Length of sequence for bert')
parser.add_argument('--trm_hidden_dim', type=int, default=50, help='Size of hidden vectors (d_model)')
parser.add_argument('--trm_num_blocks', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--trm_num_heads', type=int, default=1, help='Number of heads for multi-attention')
parser.add_argument('--trm_dropout', type=float, default=0.2, help='Dropout probability to use throughout the model')
parser.add_argument('--trm_user_dropout', type=float, default=0, help='Dropout probability to user embedding')
parser.add_argument('--trm_att_dropout', type=float, default=0.2, help='Dropout probability to use throughout the attention scores')
parser.add_argument('--trm_mask_prob', type=float, default=0.2, help='Masking probability')
parser.add_argument('--replace_prob', type=float, default=0.2, help='Replace probability')
parser.add_argument('--threshold', type=float, default=1, help='Threshold to control the coverage and confidence of query-item graph')
parser.add_argument('--item_label_replace_prob', type=float, default=0, help='Replace probability for item label')
parser.add_argument('--item_input_replace_prob', type=float, default=0, help='Replace probability for item input')
parser.add_argument('--query_input_replace_prob', type=float, default=0, help='Replace probability for query input')
parser.add_argument('--user_input_replace_prob', type=float, default=0, help='Replace probability for query input')
parser.add_argument('--graph_based_prob', type=float, default=0, help='Probability for SSE-graph')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')