import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len, num_sequences):
        self.data = np.fromfile(bin_path, dtype=np.uint16).reshape((num_sequences, seq_len))
        self.seq_len = seq_len
        self.num_sequences = num_sequences

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].astype(np.int64))

def load_tokenized_manifest(data_dir):
    manifest_path = os.path.join(data_dir, "tokenized", "manifest.json")
    with open(manifest_path) as f:
        return json.load(f)

def create_dataloader(data_dir, split="train", batch_size=16, shuffle=True):
    manifest = load_tokenized_manifest(data_dir)
    split_info = manifest[split]
    bin_path = os.path.join(data_dir, "tokenized", split_info["file"])
    seq_len = split_info["seq_len"]
    num_seq = split_info["num_sequences"]

    dataset = TokenDataset(bin_path, seq_len, num_seq)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
