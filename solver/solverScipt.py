import torch

from model import SudokuLightning
from solver.ConstraintPropogation import SudokuBoard

board = [1,6,5,2,9,3,0,0,4,0,0,0,0,0,1,6,3,2,0,2,3,0,6,0,0,9,0,0,0,9,1,7,5,0,0,0,5,0,0,9,0,0,0,1,8,0,0,2,0,3,0,0,4,9,0,9,8,0,0,0,0,0,6,0,0,0,0,0,0,9,5,0,0,0,0,4,2,9,3,8,1]
# board_array = torch.tensor(board, dtype=torch.float32).view(1, 1, 9, 9)
# ckpt_path = "C:\\Users\\samgr\\PycharmProjects\\sudoku-cp-ai\\solver\\row_ckpt.ckpt"

# model = SudokuLightning.load_from_checkpoint(ckpt_path)
# model.eval()
#
# with torch.no_grad():
#     logits = model(board_array)
#     preds = logits.view(-1).argmax()
#     print("Next cell to pick: ", preds.item())
#
# row, col = divmod(preds.item(), 9)
# print(f"Next cell coordinates: row {row}, col {col}")

sdb = SudokuBoard("board.csv")

sdb.print()
