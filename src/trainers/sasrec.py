from .base import AbstractTrainer
from .utils import Ranker, SampleRanker

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SASRecAllTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq, embed_only=False):
        super().__init__(args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq, embed_only)
        self.ce = nn.CrossEntropyLoss()
        self.ranker = Ranker(self.metric_ks, self.user2seq)

    @classmethod
    def code(cls):
        return 'sasrec_all'

    def calculate_loss(self, batch):
        users, tokens, queries, types, candidates = batch
        x, loss = self.model(tokens, queries=queries, candidates=candidates, mode="train", types=types, users=users)  # scores, loss
        return loss

    def calculate_metrics(self, batch, mode):
        users, seqs, queries, types, labels, lengths = batch
        scores = self.model(seqs, queries=queries, length=lengths, mode="all", types=types, users=users)  # B x T x C
        res = self.ranker(scores, labels, users=users, lengths=lengths)
        metrics = {}
        for i, k in enumerate(self.args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        return metrics


class SASRecSampleTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq, embed_only=False):
        super().__init__(args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq, embed_only)
        self.ce = nn.CrossEntropyLoss()
        self.ranker = SampleRanker(self.metric_ks, self.user2seq)

    @classmethod
    def code(cls):
        return 'sasrec_sample'

    def calculate_loss(self, batch):
        users, tokens, queries, types, candidates = batch
        x, loss = self.model(tokens, queries=queries, candidates=candidates, mode="train", types=types, users=users)  # scores, loss
        return loss

    def calculate_metrics(self, batch, mode):
        users, seqs, queries, types, candidates, length = batch
        scores = self.model(seqs, queries=queries, candidates=candidates, length=length, mode="sample", types=types, users=users)  # B x T x C
        res = self.ranker(scores, lengths=length)
        metrics = {}
        for i, k in enumerate(self.args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        return metrics
