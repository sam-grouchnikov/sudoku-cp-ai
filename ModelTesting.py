import torch
import pandas as pd

from solver.SudokuBoardSolver import SudokuBoard

# sudoku_file = "C:\\Users\\samgr\\PycharmProjects\\sudoku-cp-ai\\sudoku.csv"
sudoku_file = "/home/sam/sudoku/sudoku.csv"
df = pd.read_csv(sudoku_file, nrows=20000)
boards = df.iloc[:, 0].astype(str).tolist()

num_incorrect = 0

# board_temp = boards[0]
# sb = SudokuBoard(board_temp)
# pred = sb.predict(sb.getBoard())
# if sb.getValue(pred[0], pred[1]) != 0:
#    sb.print()
#    print(pred)

for board in boards:
    sb = SudokuBoard(board)
    pred = sb.predict(sb.getBoard())
    if sb.getValue(pred[0], pred[1]) != 0:
        num_incorrect += 1

print("Num incorrect:", num_incorrect)