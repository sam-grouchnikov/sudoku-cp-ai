# import pandas as pd
# import numpy as np
# import torch
# import copy
#
# from CNNGuider.model import SudokuLightning
#
#
# class SudokuBoard:
#     def __init__(self, path):
#         self.path = path
#         self.boardDF = pd.read_csv(path, header=None)
#         self.board = self.boardDF.fillna(0).astype(int).values.tolist()
#         self.domainStore = [[[1 for _ in range(9)] for _ in range(9)] for _ in range(9)]
#         self.initializeDomains()
#
#
#         ckpt_path = "C:\\Users\\samgr\\PycharmProjects\\sudoku-cp-ai\\solver\\row_ckpt.ckpt"
#         self.model = SudokuLightning.load_from_checkpoint(ckpt_path)
#
#     def predict(self, board):
#         self.model.eval()
#         board_array = torch.tensor(board, dtype=torch.float32).view(1, 1, 9, 9)
#
#         with torch.no_grad():
#             logits = self.model(board_array)
#             preds = logits.view(-1).argmax()
#             print("Next cell to pick: ", preds.item())
#
#         row, col = divmod(preds.item(), 9)
#         return [row, col]
#
#     def propagate(self):
#         for i in range(9):
#             self.propagateRows(i)
#             self.propagateCols(i)
#             self.propagateGrids(i)
#
#     def propagateRows(self, rowIdx):
#         filledVals = []
#
#         for colIdx in range(9):
#             val = self.board[rowIdx][colIdx]
#             if val != 0:
#                 filledVals.append(val)
#
#         for colIdx in range(9):
#             cellVal = self.board[rowIdx][colIdx]
#
#             if cellVal != 0:
#                 for i in range(9):
#                     self.domainStore[rowIdx][colIdx][i] = 1 if i == cellVal - 1 else 0
#             else:
#                 for i in range(9):
#                     if (i + 1) in filledVals:
#                         self.domainStore[rowIdx][colIdx][i] = 0
#
#     def initializeDomains(self):
#         for r in range(9):
#             for c in range(9):
#                 val = self.board[r][c]
#                 if val != 0:
#                     for d in range(9):
#                         self.domainStore[r][c][d] = 0
#                     self.domainStore[r][c][val - 1] = 1
#
#             self.propagateRows(r)
#             self.propagateCols(r)
#             self.propagateGrids(r)
#
#     # Propagate based on filled columns
#     def propagateCols(self, colIdx):
#         filledVals = []
#
#         for rowIdx in range(9):
#             val = self.board[rowIdx][colIdx]
#             if val != 0:
#                 filledVals.append(val)
#
#         for rowIdx in range(9):
#             cellVal = self.board[rowIdx][colIdx]
#
#             if cellVal != 0:
#                 for i in range(9):
#                     self.domainStore[rowIdx][colIdx][i] = 1 if i == cellVal - 1 else 0
#             else:
#                 for i in range(9):
#                     if (i + 1) in filledVals:
#                         self.domainStore[rowIdx][colIdx][i] = 0
#
#     # def extractFeatures(self):
#     #     # Per cell: domain store + 0/1 for empty/filled
#
#
#     # Propagate based on 3x3 grids
#     def propagateGrids(self, gridIdx):
#         startRow = (gridIdx // 3) * 3
#         startCol = (gridIdx % 3) * 3
#
#         filledVals = []
#
#         for r in range(startRow, startRow + 3):
#             for c in range(startCol, startCol + 3):
#                 val = self.board[r][c]
#                 if val != 0:
#                     filledVals.append(val)
#
#         for r in range(startRow, startRow + 3):
#             for c in range(startCol, startCol + 3):
#                 cellVal = self.board[r][c]
#
#                 if cellVal != 0:
#                     for i in range(9):
#                         self.domainStore[r][c][i] = 1 if i == cellVal - 1 else 0
#                 else:
#                     for i in range(9):
#                         if (i + 1) in filledVals:
#                             self.domainStore[r][c][i] = 0
#
#
#     def search(self, board, domainStore):
#         # if solved: return solution
#         solved = True
#         for row in board:
#             for value in row:
#                 if value == 0:
#                     solved = False
#
#         if solved:
#             return board
#
#         nextCell = self.predict(board)
#         domainStore = domainStore[nextCell[0]][nextCell[1]]
#
#         for selection in domainStore:
#
#
#         # if not: branch in domain store of current cell
#         # propagate all constraints after each selection is made
#         print(domainStore)
#         return None
#
#     def print(self):
#         print("Board:")
#         print("    " + " ".join(f"C{c}" for c in range(9)))
#         for r in range(9):
#             row_vals = " ".join(str(self.board[r][c]) for c in range(9))
#             print(f"R{r}  {row_vals}")
#
#         print("\nDomain Store:")
#         for r in range(9):
#             print(f"\nRow R{r}:")
#             for c in range(9):
#                 domain_vec = self.domainStore[r][c]
#                 possible_vals = [str(i + 1) for i in range(9) if domain_vec[i] == 1]
#
#                 possible_str = (
#                     "{" + ",".join(possible_vals) + "}"
#                     if possible_vals else "-"
#                 )
#
#                 print(f"  Cell (R{r},C{c}): {domain_vec}  ->  {possible_str}")