import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len, n_seqs):
        self.data = np.fromfile(bin_path, dtype=np.uint16).reshape((n_seqs, seq_len))
        self.n_seqs = n_seqs

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].astype(np.int64))


def load_manifest(data_dir):
    path = os.path.join(data_dir, "tokenized", "manifest.json")
    with open(path) as f:
        return json.load(f)


def create_dataloader(data_dir, split="train", batch_size=16, shuffle=True):
    manifest = load_manifest(data_dir)
    info = manifest[split]
    bin_path = os.path.join(data_dir, "tokenized", info["file"])
    ds = TokenDataset(bin_path, info["seq_len"], info["num_sequences"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False, drop_last=True)


load_tokenized_manifest = load_manifest
