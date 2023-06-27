import torch
from torch.utils.data.sampler import (SequentialSampler,
                                        RandomSampler,
                                        SubsetRandomSampler,
                                        WeightedRandomSampler,
                                        BatchSampler)
import random 
a = [0,1,2,3,4,5,6,7,8,9]
b = BatchSampler(a,3,drop_last=True)

for i in b:
    print(i)
