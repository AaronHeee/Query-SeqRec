from .base import AbstractNegativeSampler

from tqdm import trange
from tqdm import tqdm

import numpy as np
import random

TEST_MAX = 1000000
THRESHOLD = 1000000

class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self, mode='val'):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}

        print('Sampling negative items')

        # fast sampling --> assuming that the prob of sampling postive items from a large items pool is 0
        candidates = range(self.item_count)
        if mode == 'train':
            data = self.train
        elif mode == 'val':
            data = self.val
        elif mode == 'test':
            data = self.test
        data_keys = list(data.keys())
        random.shuffle(data_keys)

        for i, user in tqdm(enumerate(data_keys)):
            if i > TEST_MAX: 
                break
            negative_samples[user] = random.choices(candidates, k=self.sample_size+1)
            if data[user]['click'][0][-1] in negative_samples[user]:
                negative_samples[user].remove(data[user]['click'][0][-1])
            else:
                negative_samples[user] = negative_samples[user][:-1]

        return negative_samples
