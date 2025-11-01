from solver.SudokuBoardSolver import SudokuBoard

sdb = SudokuBoard("board.csv")

sdb.print()
solution = sdb.search()
for row in solution:
    print(row)