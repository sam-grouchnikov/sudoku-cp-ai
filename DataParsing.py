import pandas as pd
import numpy as np

from ConstraintPropogation import SudokuBoard

df = pd.read_csv("board.csv", header=None)
board = df.fillna(0).values.tolist()

cp_board = np.ones((9, 9, 9), dtype=int)

sboard = SudokuBoard("board.csv")

sboard.propagateRows(0)

sboard.print()