import math
import os
import re
import time
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from PoemDataset import PoemDataset
from model import TransformerModel
# torchtext.disable_torchtext_deprecation_warning()
DATASET_PATH = 'poem_dataset_final/poem_final.csv'
MAX_SEQ_LEN = 25
EMBEDDING_DIMS = 128
HIDDEN_DIMS = 128
N_LAYERS = 2
N_HEADS = 4
DROPOUT = 0.2
LR = 5.0
EPOCHS = 100
#################### get vocab ###########################
def text_normalize(text): # khong can xoa di dau cau hay doi chu hoa thanh chu thuong
    text = text.strip() # Xoa khoang trang dau cuoi
    return text

def tokenizer(text):
    return text.split()

def yield_tokens(df):
    for idx, row in df.iterrows():
        yield tokenizer(row['content'])

def build_vocab(df):
    df['content'] = df['content'].apply(lambda x: text_normalize(x))
    vocab = build_vocab_from_iterator(
        yield_tokens(df),
        specials=['<unk>', '<pad>', '<sos>', '<eos>', '<eol>']
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab

#################### vectorize ###########################
def pad_and_truncate(input_ids, max_seq_len):
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
    else:
        input_ids += [PAD_TOKEN] * (max_seq_len - len(input_ids))

    return input_ids

def vectorize(text, max_seq_len):
    input_ids = [vocab[token] for token in tokenizer(text)]
    input_ids = pad_and_truncate(input_ids, max_seq_len)

    return input_ids

def decode(input_ids, vocab):
    return [vocab.get_itos()[token_id] for token_id in input_ids]

#################### train ###########################
def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for idx, samples in enumerate(train_loader):
            input_seqs, target_seqs, padding_masks = samples
            input_seqs = input_seqs.to(device)
            target_seqs = target_seqs.to(device)
            padding_masks = padding_masks.to(device).permute(1, 0)

            output = model(input_seqs, padding_mask=padding_masks)
            output = output.permute(0, 2, 1)
            loss = criterion(output, target_seqs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            losses.append(loss.item())

        total_loss = sum(losses) / len(losses)
        print(f'EPOCH {epoch+1}\tLoss {total_loss}')
        scheduler.step()

    torch.save(model.state_dict(), 'model_poem_gen_1.pth')

df = pd.read_csv(DATASET_PATH)
vocab = build_vocab(df)
VOCAB_SIZE = len(vocab)
#################### main ###########################
if __name__ == '__main__':
    df = pd.read_csv(DATASET_PATH)
    vocab = build_vocab(df)
    PAD_TOKEN = vocab['<pad>']
    EOS_TOKEN = vocab['<eos>']
    
    TRAIN_BS = 256

    train_dataset = PoemDataset(
        df=df,
        tokenizer=tokenizer,
        vectorizer=vectorize,
        max_seq_len=MAX_SEQ_LEN,
        PAD_TOKEN=PAD_TOKEN
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BS,
        shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tests = torch.randint(1, 10, (1, 10)).to(device)

    model = TransformerModel(
        VOCAB_SIZE,
        EMBEDDING_DIMS,
        N_HEADS,
        HIDDEN_DIMS,
        N_LAYERS,
        DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    # train(model, train_loader, criterion, optimizer, scheduler, device)