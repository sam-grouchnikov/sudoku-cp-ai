import pandas as pd
import numpy as np

from solver.ConstraintPropogation import SudokuBoard

df = pd.read_csv("solver/board.csv", header=None)
board = df.fillna(0).values.tolist()

cp_board = np.ones((9, 9, 9), dtype=int)

sboard = SudokuBoard("solver/board.csv")

sboard.print()