import os
import math
import numpy as np
import json

import torch
from torch import nn as nn
import torch.nn.functional as F

from src.utils.utils import fix_random_seed_as

MAX_VAL = 1e8
MIN_VAL = 1e-8

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class Attention(nn.Module):
    def __init__(self, hidden, max_len, is_relative=False):
        super().__init__()

    "Compute 'Scaled Dot Product Attention"
    def forward(self, query, key, value, mask=None, dropout=None, weight=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 1, -MAX_VAL)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class MultiheadAttention(nn.Module):
    "Take in model size and number of heads."
    def __init__(self, d_model, h, dropout=0.1, max_len=None, args=None):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.args = args

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model, max_len)
        self.attention = Attention(d_model * h, max_len, True)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attn_mask=None, external=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                                 for l, x in zip(self.linear_layers[:3], (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=attn_mask, dropout=self.dropout, weight=external)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn


class SASRecModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb_device_idx = self.args.emb_device_idx
        fix_random_seed_as(args.model_init_seed)

        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.CrossEntropyLoss()

        self.num_items = args.num_items
        self.num_tokens = args.num_tokens
        self.num_words = args.num_words
        self.num_users = args.num_users

        self.item_emb = torch.nn.Embedding(self.num_tokens+2, args.trm_hidden_dim, padding_idx=-1)
        self.pad_token = self.item_emb.padding_idx
        self.type_emb = torch.nn.Embedding(2, args.trm_hidden_dim)
        self.pos_emb = torch.nn.Embedding(args.trm_max_len, args.trm_hidden_dim) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.trm_dropout, inplace=True)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)

        for _ in range(args.trm_num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  MultiheadAttention(args.trm_hidden_dim,
                                                 args.trm_num_heads,
                                                 args.trm_dropout,
                                                 args.trm_max_len,
                                                 self.args)

            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.trm_hidden_dim, args.trm_dropout)
            self.forward_layers.append(new_fwd_layer)
        
        # weights initialization
        self.init_weights()
        self.item_emb.weight.data[-1] *= 0 # set padding embedding as 0
        self.external = None

    def log2feats(self, log_seqs, queries=None, attention_mask=None, users=None, types=None):
        seqs = self.lookup(log_seqs)

        if len(seqs.size()) > 3: # bow model
            seqs = seqs.sum(dim=-2) / ((log_seqs != self.pad_token).sum(dim=-1, keepdim=True) + 1e-12)
            log_seqs = log_seqs[..., 0]

        if self.args.early_query:
            aug_queries = queries + self.num_items
            seqs_q = self.lookup(aug_queries)
            if len(seqs_q.size()) > 3: # bow model
                seqs_q = seqs_q.sum(dim=-2) / ((aug_queries != self.pad_token).sum(dim=-1, keepdim=True) + 1e-12)
            seqs = seqs_q + seqs

        external_sim = None

        positions = torch.arange(seqs.shape[1]).long().unsqueeze(0).repeat([seqs.shape[0], 1])
        seqs = seqs + self.pos_emb(positions.to(seqs.device)) + self.type_emb(types) 
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == self.pad_token).bool()

        if attention_mask is None:
            attention_mask = (~torch.tril(torch.ones((seqs.size(1), seqs.size(1)), dtype=torch.bool, device=seqs.device)).unsqueeze(0)) | (timeline_mask.unsqueeze(-1))
        attention_mask = attention_mask.unsqueeze(1)

        attn_output_weights = []

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_output_weight = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            
            attn_output_weights.append(attn_output_weight)
            
            seqs = mha_outputs
            # seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs) + seqs

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats, attn_output_weights


    def forward(self, x, queries=None, candidates=None, length=None, save_name=None, mode="train", users=None, need_weights=False, types=None):

        idx1, idx2 = self.select_predict_index(types) if mode == "train" else (torch.arange(x.size(0)), length.squeeze())
        log_feats, attn_weights = self.log2feats(x, queries, users=users, types=(types==0).long())

        log_feats = log_feats[idx1, idx2]

        if mode == "serving":
            if need_weights:
                return log_feats, attn_weights
            return log_feats

        elif mode == "train":
            candidates = candidates[idx1, idx2]
            pos_seqs = candidates[:, 0]

            if self.args.train_negative_sample_size <= 0:
                logits = self.all_predict(log_feats) # all predicts
                labels = pos_seqs
            else:
                candidates = torch.cat((pos_seqs.unsqueeze(-1), torch.randint(0, self.num_items, size=(pos_seqs.size(0), self.args.train_negative_sample_size)).to(pos_seqs.device)), dim=-1)
                w = self.lookup(candidates).transpose(2,1)
                log_feats = log_feats.unsqueeze(1)
                logits = torch.bmm(log_feats, w).squeeze(1)
                labels = torch.zeros_like(pos_seqs, device=pos_seqs.device)

            # softmax #            
            loss = self.loss(logits, labels)

            return logits, loss 

            # softmax #

        else:

            if candidates is not None:
                log_feats = log_feats.unsqueeze(1) # x is (batch_size, 1, embed_size)
                w = self.lookup(candidates).transpose(2,1) # (batch_size, embed_size, candidates)
                logits = torch.bmm(log_feats, w).squeeze(1) # (batch_size, candidates)
            else:
                
                logits = self.all_predict(log_feats) # test in items candidates
            if need_weights:
                return logits, attn_weights
            return logits
        

    def select_predict_index(self, types):

        # set query_session_id: 
        # E.g. (cat, 1, paper, 2, 3)  —> 
        #   <(cat), 1, q->i>, <(cat, 1), paper, i->q>, <(cat, 1, paper), 2, q->i>, <(cat, 1, paper, 2), 3, i->i> , 
        #   where token is (0, 2, 0, 1, 2)
        #   we mark i->q and pad_token as 2, which will be removed by this function

        return (types<2).nonzero(as_tuple=True)            

    def init_weights(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if ('norm' not in n) and ('bias' not in n):
                    try:
                        torch.nn.init.xavier_uniform_(p.data)
                    except:
                        pass # just ignore those failed init layers

    def all_predict(self, log_feats):
        if self.args.emb_device_idx is None:
            w = self.item_emb.weight.transpose(1,0)
            res = torch.matmul(log_feats, w)
        elif type(self.args.emb_device_idx) == str and self.args.emb_device_idx.lower() == 'cpu':
            w = self.item_emb.weight.transpose(1,0)
            res = torch.matmul(log_feats.to('cpu'), w).to(self.args.device)
        else:
            res = 0
            for i, emb_device in enumerate(self.emb_device_idx):
                b, e = self.emb_device_idx[emb_device]
                x = log_feats[..., b:e].to(emb_device)
                res += torch.matmul(x, self.item_emb_list[i].weight.transpose(1,0)).to(self.args.device)
        return res[:, :self.num_items]

    def lookup(self, x):
        if self.args.emb_device_idx is None:
            return self.item_emb(x)
        elif type(self.args.emb_device_idx) == str and self.args.emb_device_idx.lower() == 'cpu':
            return self.item_emb(x.to('cpu')).to(self.args.device)
        else:
            res = []
            for emb_layer in self.item_emb_list:
                device = emb_layer.weight.device
                res.append(emb_layer(x.to(device)).to(self.args.device))
            return torch.cat(res, dim=-1)

    def large_embed_to(self, device):
        if self.args.emb_device_idx is None:
            return self.to(device)
        elif self.args.emb_device_idx.lower() == 'cpu':
            temp = self.item_emb
            self.item_emb = None
            self.to(device)
            self.item_emb = temp
            print('move embedding layer to:', self.item_emb.weight.device)
            return self
        try:
            self.emb_device_idx = eval(self.args.emb_device_idx)
            temp = self.item_emb.weight.data.detach()
            self.item_emb = None
            self.to(device)
            self.item_emb_list = []
            for emb_device in self.emb_device_idx:
                b, e = self.emb_device_idx[emb_device]
                partial_emb = nn.Embedding.from_pretrained(temp[..., b:e], freeze=False, padding_idx=-1).to(emb_device)
                self.item_emb_list.append(partial_emb)
            self.item_emb_list = nn.ModuleList(self.item_emb_list)
            print('move embedding layer to:', self.emb_device_idx)
            return self
        except:
            print("ERROR: please follow this rule to set emb_device_idx: None / cpu / {'cpu':(0,16), 'cuda:0':(16,50)}")
            exit()

    def device_state_dict(self):
        if type(self.emb_device_idx) != dict:
            return self.state_dict()
        else:
            params = self.state_dict()
            name_from, name_to = 'item_emb_list', 'item_emb'
            temp = []
            for i in params.keys():
                j = i.split('.')
                if j[0] == name_from:
                    temp.append((int(j[1]), i, params[i].to('cpu')))
            for _, i, _ in temp:
                del params[i]
            temp = [t[-1] for t in sorted(temp)]
            params[name_to+".weight"] = torch.cat(temp, dim=-1)
            return params
