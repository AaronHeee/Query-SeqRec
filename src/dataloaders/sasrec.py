import os
import random
import torch
import torch.utils.data as data_utils
import numpy as np
import json

BOW_LEN = 5 # max length of words in query

class SASRecDataloader(object):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.train = dataset.train
        self.val = dataset.val
        self.test = dataset.test
        self.umap = dataset.umap
        self.smap = dataset.smap
        self.wmap = dataset.wmap
        self.num_users = args.num_users = len(self.umap)
        self.num_items = args.num_items = len(self.smap)
        self.num_words = args.num_words = len(self.wmap)
        self.num_tokens = args.num_tokens = len(self.smap) + len(self.wmap)

        self.max_len = args.trm_max_len
        self.CLOZE_MASK_TOKEN = self.num_tokens
        self.PAD_TOKEN = self.CLOZE_MASK_TOKEN + 1
        self.PAD_QUERY = self.num_words + 1

        self.negative_samples_val = dataset.negative_samples_val
        self.negative_samples_test = dataset.negative_samples_test
        

    @classmethod
    def code(cls):
        return 'sasrec'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_eval_loader(mode='val')
        test_loader = self._get_eval_loader(mode='test')
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = SASRecTrainDataset(self.train, self.max_len, 
                    self.CLOZE_MASK_TOKEN, self.num_items, self.rng, self.PAD_TOKEN, self.PAD_QUERY)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size, 
                    drop_last=False, shuffle=True, pin_memory=True)
        dataloader.pad_token = self.PAD_TOKEN
        return dataloader

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        negs = self.negative_samples_val if mode == 'val' else self.negative_samples_test
        # candidates = None
        candidates = self._get_eval_candidates(mode)
        dataset = SASRecEvalDataset(self.train, self.val, self.test, self.max_len, self.CLOZE_MASK_TOKEN, self.PAD_TOKEN, self.PAD_QUERY,
                mode=mode, is_all=(self.args.trainer_code=="sasrec_all"), negative_samples=negs, u_candidates=candidates, num_items=self.num_items)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, drop_last=False,
                shuffle=True, pin_memory=True)
        return dataloader

    def _get_eval_candidates(self, mode):
        if self.args.eval_mode is None:
            return None
        data = json.load(open(os.path.join(self.args.data_path, "%s_query.json" % mode), 'r'))
        sudden = []
        not_sudden = []

        for i in data:
            queries = set(data[i][:-1])
            target = data[i][-1]
            if target in queries:
                not_sudden.append(int(i))
            else:
                sudden.append(int(i))

        if self.args.eval_mode == 'sudden':
            return set(sudden)
        if self.args.eval_mode == 'not_sudden':
            return set(not_sudden)

class SASRecTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_token, num_items, rng, pad_token, pad_query):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.rng = rng
        self.num_items = num_items
        self.pad_token = pad_token
        self.pad_query = pad_query

    def __len__(self):
        return len(self.users)

    def _pad_to_len(self, seq, pad):
        res = []
        for s in seq:
            res_ = [pad] * BOW_LEN
            res_[:min(BOW_LEN, len(s))] = s[:min(BOW_LEN, len(s))]
            res.append(res_)
        return res

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]['click']

        labels = seq[-self.max_len-1:]
        if len(labels) > 1:
            tokens = labels[:-1]
            labels = [i[0] for i in labels[1:]]
        else:
            tokens = [[self.pad_token]]
            labels = labels[0]
        length = len(tokens)
        query = self._pad_to_len(self.u2seq[user]['query'][-length:], pad=self.pad_query)

        item_negative = []
        while len(item_negative) < length:
            item_negative_tmp = self.rng.randint(0, self.num_items-1)
            while item_negative_tmp in self.u2seq[user]:
                item_negative_tmp = self.rng.randint(0, self.num_items-1)
            item_negative.append(item_negative_tmp)

        padding_len = self.max_len - length

        tokens = np.array(self._pad_to_len(tokens, pad=self.pad_token) + [[self.pad_token] * BOW_LEN] * padding_len)
        query = query + [[self.pad_query] * BOW_LEN] * padding_len

        # set query_session_id: 
        # E.g. (cat, 1, paper, 2, 3)  —> 
        #   <(cat), 1, q->i>, <(cat, 1), paper, i->q>, <(cat, 1, paper), 2, q->i>, <(cat, 1, paper, 2), 3, i->i> , where token is (0, 2, 0, 1, 2)
        #   we mark i->q and pad_token as 2, which will be removed during training

        tokens_ = tokens[...,0]
        query_session_id = np.full(len(tokens_), 1)
        query_session_id[-padding_len:] = 2

        query_index = np.where((self.num_items <= tokens_) & (tokens_ < self.pad_token))[0]
        if len(query_index):
            end_index = query_index[1:]-1 if query_index[0] == 0 else query_index-1 
            query_session_id[end_index] = 2
            query_session_id[query_index] = 0

        try:
            labels = torch.LongTensor(labels + [-100] * padding_len).unsqueeze(-1)
        except: 
            import pdb; pdb.set_trace()
        negs = torch.LongTensor(item_negative + [-100] * padding_len).unsqueeze(-1)

        return torch.LongTensor([user]), torch.LongTensor(tokens), torch.LongTensor(query), torch.LongTensor(query_session_id), torch.cat((labels, negs), dim=-1)


class SASRecEvalDataset(data_utils.Dataset):
    def __init__(self, train, val, test, max_len, mask_token, pad_token, pad_query, mode, is_all, negative_samples=None, u_candidates=None, num_items=None):
        self.u_candidates = u_candidates
        self.u2seq, self.u2answer = self._split(train, val, test, mode)
        self.negative_samples = negative_samples
        if self.negative_samples is not None:
            self.users = {i:u for i,u in enumerate(self.negative_samples)}
        else:
            self.users = {i:u for i,u in enumerate(self.u2seq)}
        self.max_len = max_len
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.pad_query = pad_query
        self.mode = mode
        self.is_all = is_all
        self.num_items = num_items

    def _split(self, train, val, test, mode):
        session_input, session_output = {}, {}
        for u in train:
            if self.u_candidates is None or u in self.u_candidates:
                if mode == "val":
                    session_input[u] = train[u]
                    session_output[u] = val[u]
                elif mode == "test":
                    session_input[u] = {'click': train[u]['click']+val[u]['click'], 'query': train[u]['query']+val[u]['query']}
                    session_output[u] = test[u]
        return session_input, session_output

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self.all(index) if self.is_all else self.sample(index)

    def _pad_to_len(self, seq, pad):
        res = []
        for s in seq:
            res_ = [pad] * BOW_LEN
            res_[:min(BOW_LEN, len(s))] = s[:min(BOW_LEN, len(s))]
            res.append(res_)
        return res

    def all(self, index):
        user = self.users[index]
        seq = self.u2seq[user]['click']
        answer = self.u2answer[user]['click'][0]

        seq = self._pad_to_len(seq[-self.max_len:], self.pad_token)
        length = len(seq)
        query = self._pad_to_len(self.u2seq[user]['query'][-length:], self.pad_query)
        padding_len = self.max_len - length
        seq = np.array(seq + [[self.pad_token] * BOW_LEN] * padding_len)
        query = query + [[self.pad_query] * BOW_LEN] * padding_len

        seq_ = seq[...,0]
        query_session_id = np.full(len(seq_), 1)
        query_session_id[-padding_len:] = 2

        query_index = np.where((self.num_items <= seq_) & (seq_ < self.pad_token))[0]
        if len(query_index):
            end_index = query_index[1:]-1 if query_index[0] == 0 else query_index-1 
            query_session_id[end_index] = 2
            query_session_id[query_index] = 0

        if answer[0] > self.num_items:
            import pdb; pdb.set_trace()

        return torch.LongTensor([user]), torch.LongTensor(seq), torch.LongTensor(query), torch.LongTensor(query_session_id), torch.LongTensor(answer), torch.LongTensor([length-1])

    def sample(self, index):
        user = self.users[index]
        seq = self.u2seq[user]['click']
        answer = self.u2answer[user]['click'][0]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = self._pad_to_len(seq[-self.max_len:], self.pad_token)
        length = len(seq)
        query = self._pad_to_len(self.u2seq[user]['query'][-length:], self.pad_query)
        padding_len = self.max_len - length
        seq = np.array(seq + [[self.pad_token] * BOW_LEN] * padding_len)
        query = query + [[self.pad_query] * BOW_LEN] * padding_len

        seq_ = seq[...,0]
        query_session_id = np.full(len(seq_), 1)
        query_session_id[-padding_len:] = 2

        query_index = np.where((self.num_items <= seq_) & (seq_ < self.pad_token))[0]
        if len(query_index):
            end_index = query_index[1:]-1 if query_index[0] == 0 else query_index-1 
            query_session_id[end_index] = 2
            query_session_id[query_index] = 0

        return torch.LongTensor([user]), torch.LongTensor(seq), torch.LongTensor(query), torch.LongTensor(query_session_id), torch.LongTensor(candidates), torch.LongTensor([length-1])

