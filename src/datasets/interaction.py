from .negative_samplers import negative_sampler_factory

import os
import json
import random

class ItemQueryDataset(object):
    def __init__(self, args):
        self.path = args.data_path
        self.train = self.read_json(os.path.join(self.path, "train.json"), True)
        if 'aug_sasrec' in args.dataloader_code:
             self.train_triplets = self.read_json(os.path.join(self.path, "train_triplets_str_alpha_%.1f.json" % args.threshold), False)
             print("load ... train_triplets_str_alpha_%.1f.json" % args.threshold)
        self.val = self.read_json(os.path.join(self.path, "val.json"), True)
        self.test = self.read_json(os.path.join(self.path, "test.json"), True)
        self.data = self.merge(self.train, self.val, self.test)
        self.umap = self.read_json(os.path.join(self.path, "umap.json"))
        self.smap = self.read_json(os.path.join(self.path, "smap.json"))
        self.wmap = self.read_json(os.path.join(self.path, "wmap.json"))
        if 'sample' in args.trainer_code:
            if args.test_negative_sampler_code == 'random':
                negative_sampler = negative_sampler_factory(args.test_negative_sampler_code, self.train, self.val, self.test,
                                                            len(self.umap), len(self.smap),
                                                            args.test_negative_sample_size,
                                                            args.test_negative_sampling_seed,
                                                            self.path)
                self.negative_samples_val = negative_sampler.get_negative_samples(mode='val')
                self.negative_samples_test = negative_sampler.get_negative_samples(mode='test')
            elif args.test_negative_sampler_code == 'vse':
                print("loading vse re-ranking results ...")
                self.negative_samples_val = self.read_json(os.path.join(self.path, "val_vse.json"), True)
                for u in self.negative_samples_val:
                    self.negative_samples_val[u] = self.negative_samples_val[u][: args.test_negative_sample_size]
                self.negative_samples_test = self.read_json(os.path.join(self.path, "test_vse.json"), True)
                for u in self.negative_samples_test:
                    self.negative_samples_test[u] = self.negative_samples_test[u][: args.test_negative_sample_size]
        else:
            self.negative_samples_val = None
            self.negative_samples_test = None

    def merge(self, a, b, c):
        data = {}
        for i in a:
            data[i] = a[i]
        for i in b:
            data[i] = b[i]
        for i in c:
            data[i] = c[i]
        return data

    def read_json(self, path, as_int=False):
        with open(path, 'r') as f:
            raw = json.load(f)
            if as_int:
                data = dict((int(key), value) for (key, value) in raw.items())
            else:
                data = dict((key, value) for (key, value) in raw.items())
            del raw
            return data
    
    @classmethod
    def code(cls):
        return "item_query"
