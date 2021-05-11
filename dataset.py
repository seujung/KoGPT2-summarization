import os
import glob
import torch
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader

MASK = '<unused0>'
SUMMARY = '<unused1>'
BOS = '</s>'
EOS = '</s>'
PAD = '<pad>'

class KoGPTSummaryDataset(Dataset):
    def __init__(self, file, tok, max_len,
                 bos_token=BOS, eos_token=EOS,
                 pad_token=PAD, mask_token=MASK,
                 summary_token = SUMMARY,
                 ignore_index = -100
                ):
        super().__init__()
        self.tok = tok
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep='\t')
        self.len = self.docs.shape[0]
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.summary_token = summary_token
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs, pad_index):
        if len(inputs) < self.max_len:
            pad = [pad_index] *(self.max_len - len(inputs))
            inputs = inputs + pad
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        article = self.tok.encode(self.bos_token) + self.tok.encode(instance['news'])
        len_article = len(article)
        
        summary = self.tok.encode(self.summary_token) + self.tok.encode(instance['summary']) + self.tok.encode(self.eos_token)
        len_summary = len(summary)
        context = article + summary

        if len(context) > self.max_len:
            additional_len = len(context) - self.max_len
            article = article[:-additional_len]
            len_article = len(article)
            context = article + summary

        labels = [-100] * len_article + summary[1:]
        mask = [0] * len_article + [1] * len_summary + [0] * (self.max_len - len_article - len_summary)

        if len(context) < self.max_len:
            context = self.add_padding_data(context, self.tok.pad_token_id)

        if len(labels) < self.max_len:
            labels = self.add_padding_data(labels, -100)

        return {'input': np.array(context, dtype=np.int_),
                'mask': np.array(mask, dtype=np.int_),
                'label': np.array(labels, dtype=np.int_)}

    def __len__(self):
        return self.len

