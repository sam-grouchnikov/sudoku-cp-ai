from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import numpy as np

from PreprocessData import preprocess
from solver.SudokuBoardSolver import SudokuBoard


class SudokuDataset(Dataset):
    def __init__(self, file):
        self.data = pd.read_csv(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        board_str = row["board"]
        target_idx = int(row["label"])
        sb = SudokuBoard(board_str)
        domainStore = sb.getDomainStore()

        x = preprocess(board_str, domainStore)
        y = torch.tensor(target_idx, dtype=torch.long)

        return x, y
