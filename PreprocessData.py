import numpy as np
import torch
import pandas as pd


# Load CSV (keep only column 0)
# df = pd.read_csv("sudoku.csv", header=None)
#
# # Extract the column with board strings
# boards = df.iloc[:, 0]
#
# # Count zeros in each board
# zero_counts = boards.apply(lambda s: s.count("0"))
#
# # Build new dataframe
# out = pd.DataFrame({
#     "board": boards,
#     "zero_count": zero_counts
# })
#
# # Sort ascending by #zeros
# out = out.sort_values("zero_count").reset_index(drop=True)
#
# # Save
# out.to_csv("sudoku_boards_sorted.csv", index=False)

# df = pd.read_csv("sudoku_boards_sorted.csv")
#
# # Group by zero_count
# groups = df.groupby("zero_count")


# rows = []

# for zero_count, group in groups:
#     if len(group) <= 2000:
#         # Take all examples
#         rows.append(group)
#     else:
#         # Take the first 200
#         rows.append(group.iloc[:2000])
#
# # Combine all collected rows
# sampled = pd.concat(rows, ignore_index=True)
#
# # Save to a new file
# sampled.to_csv("sudoku_board_samples.csv", index=False)
#
#
# def preprocess(board_str, domainStore=None):
#     board = np.array(list(map(int, board_str)), dtype=np.float32).reshape(9, 9)
#     features = np.zeros((21, 9, 9), dtype=np.float32)
#
#     for r in range(9):
#         for c in range(9):
#             val = int(board[r][c])
#             if 1 <= val <= 9:
#                 features[val - 1][r][c] = 1.0
#
#     features[9] = (board == 0).astype(np.float32)
#
#     if domainStore is not None:
#         for r in range(9):
#             for c in range(9):
#                 features[10, r, c] = np.sum(domainStore[r][c]) / 9.0
#     else:
#         features[10] = 0.0
#
#     if domainStore is not None:
#         features = np.concatenate([features, np.array(domainStore).transpose(2, 0, 1)], axis = 0)
#
#     return torch.tensor(features, dtype=torch.float32)
#
# def getDomainStore(board_string: str):
#
#     # Convert to board (9x9 integers)
#     flat_array = np.array([int(ch) for ch in board_string])
#     board = flat_array.reshape(9, 9).tolist()
#
#     # Initialize all domains as fully open (1–9 possible)
#     domainStore = [[[1 for _ in range(9)] for _ in range(9)] for _ in range(9)]
#
#     def propagateRows(r):
#         filledVals = [board[r][c] for c in range(9) if board[r][c] != 0]
#         for c in range(9):
#             val = board[r][c]
#             if val != 0:
#                 for i in range(9):
#                     domainStore[r][c][i] = 1 if i == val - 1 else 0
#             else:
#                 for i in range(9):
#                     if (i + 1) in filledVals:
#                         domainStore[r][c][i] = 0
#
#     def propagateCols(c):
#         filledVals = [board[r][c] for r in range(9) if board[r][c] != 0]
#         for r in range(9):
#             val = board[r][c]
#             if val != 0:
#                 for i in range(9):
#                     domainStore[r][c][i] = 1 if i == val - 1 else 0
#             else:
#                 for i in range(9):
#                     if (i + 1) in filledVals:
#                         domainStore[r][c][i] = 0
#
#     def propagateGrids(g):
#         startRow, startCol = (g // 3) * 3, (g % 3) * 3
#         filledVals = [board[r][c]
#                       for r in range(startRow, startRow + 3)
#                       for c in range(startCol, startCol + 3)
#                       if board[r][c] != 0]
#         for r in range(startRow, startRow + 3):
#             for c in range(startCol, startCol + 3):
#                 val = board[r][c]
#                 if val != 0:
#                     for i in range(9):
#                         domainStore[r][c][i] = 1 if i == val - 1 else 0
#                 else:
#                     for i in range(9):
#                         if (i + 1) in filledVals:
#                             domainStore[r][c][i] = 0
#
#     # Initialization propagation
#     for r in range(9):
#         propagateRows(r)
#         propagateCols(r)
#         propagateGrids(r)
#
#     return np.array(domainStore, dtype=np.uint8)

# board = "561092730020780090900005046600000427010070003073000819035900670700103080000000050"
# ds = getDomainStore(board)
# data_tensor = preprocess(board, ds)
