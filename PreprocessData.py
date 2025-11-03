import numpy as np
import torch

from solver.SudokuBoardSolver import SudokuBoard


def preprocess(board_str, domainStore=None):
    board = np.array(list(map(int, board_str)), dtype=np.float32).reshape(9, 9)
    features = np.zeros((21, 9, 9), dtype=np.float32)

    for r in range(9):
        for c in range(9):
            val = int(board[r][c])
            if 1 <= val <= 9:
                features[val - 1][r][c] = 1.0

    features[9] = (board == 0).astype(np.float32)

    if domainStore is not None:
        for r in range(9):
            for c in range(9):
                features[10, r, c] = np.sum(domainStore[r][c]) / 9.0
    else:
        features[10] = 0.0

    if domainStore is not None:
        features = np.concatenate([features, np.array(domainStore).transpose(2, 0, 1)], axis = 0)

    return torch.tensor(features, dtype=torch.float32)

board = "561092730020780090900005046600000427010070003073000819035900670700103080000000050"
sb = SudokuBoard(board)
domainStore = sb.getDomainStore()
data_tensor = preprocess(board, domainStore)
print(data_tensor)