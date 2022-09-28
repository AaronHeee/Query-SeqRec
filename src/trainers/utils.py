import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

MAX_VAL = 1e4
THRESHOLD = 0.5

class Ranker(nn.Module):
    def __init__(self, metrics_ks, user2seq):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        self.user2seq = user2seq

    def forward(self, scores, labels, lengths=None, seqs=None, users=None):

        labels = labels.squeeze(-1)
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        if seqs is not None:
            scores[torch.arange(scores.size(0)).unsqueeze(-1), seqs] = -MAX_VAL # mask the rated items

        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )

        if scores.size(0) == 1:
            with open("u-sasrec-details.csv", "a") as f:
                 f.write("%d,%d," % (users.item(), lengths.item()) + ",".join([str(i) for i in res]) +"\n") # user id, length, n@5, hr@5, n@10, hr@10, n@20, hr@20 

        return res 

class SampleRanker(nn.Module):
    def __init__(self, metrics_ks, user2seq):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        self.user2seq = user2seq

    def forward(self, scores):
        predicts = scores[:, 0].unsqueeze(-1) # gather perdicted values
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )
        return res
