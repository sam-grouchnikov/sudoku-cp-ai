from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import numpy as np

class SudokuDataset(Dataset):
    def __init__(self, file, num_features = 1, use_one_hot_target = True):
        self.data = pd.read_csv(file)
        self.num_features = num_features
        self.use_one_hot_target = use_one_hot_target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        features_flat = row[:-1].values.astype(np.float32)
        features = features_flat.reshape(81, self.num_features)
        features = features.reshape(9, 9, self.num_features)
        features = np.transpose(features, (2, 0, 1))

        x = torch.tensor(features, dtype=torch.float32)

        target_idx = int(row[-1])
        y = torch.zeros(9, 9, dtype=torch.float32)
        row_idx = target_idx // 9
        col_idx = target_idx % 9
        y[row_idx, col_idx] = 1.0

        return x, y
