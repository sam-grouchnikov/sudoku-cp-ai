import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

from solver.SudokuBoardSolver import SudokuBoard

# sudoku_file = "C:\\Users\\samgr\\PycharmProjects\\sudoku-cp-ai\\sudoku-3m.csv"
sudoku_file = "/home/sam/sudoku/sudoku-3m.csv"
df = pd.read_csv(sudoku_file, nrows=2000)
boards = df.iloc[:, 1].astype(str).tolist()

board_sorted = [[] for _ in range(81)]

for board in boards:
    new_board = board.replace(".", "0")
    count = new_board.count("0")
    board_sorted[count].append(new_board)

times_cnn = [0 for _ in range(81)]
branches_cnn = [0 for _ in range(81)]

for index, boardSet in enumerate(board_sorted):
    if len(boardSet) == 0:
        continue
    total_solver_timed = 0
    total_solver_branches = 0
    for board in boardSet:
        sdb = SudokuBoard(board)
        start = time.time()
        sol, branches = sdb.search("cnn")
        end = time.time()
        total_time = end - start
        total_solver_timed += total_time
        total_solver_branches += sdb.callCount()

    print("Finished index ", index)
    times_cnn[index] = total_solver_timed / len(boardSet)
    branches_cnn[index] = total_solver_branches / len(boardSet)

print("CNN done")

times_hybrid = [0 for _ in range(81)]
branches_hybrid = [0 for _ in range(81)]

for index, boardSet in enumerate(board_sorted):
    if len(boardSet) == 0:
        continue
    total_solver_timed = 0
    total_solver_branches = 0
    for board in boardSet:
        sdb = SudokuBoard(board)
        start = time.time()
        sol, branches = sdb.search("hybrid")
        end = time.time()
        total_time = end - start
        total_solver_timed += total_time
        total_solver_branches += sdb.callCount()

    print("Finished index ", index)
    times_hybrid[index] = total_solver_timed / len(boardSet)
    branches_hybrid[index] = total_solver_branches / len(boardSet)

print("Hybrid done")

times_naive = [0 for _ in range(81)]
branches_naive = [0 for _ in range(81)]

for index, boardSet in enumerate(board_sorted):
    if len(boardSet) == 0:
        continue
    total_solver_timed = 0
    total_solver_branches = 0
    for board in boardSet:
        sdb = SudokuBoard(board)
        start = time.time()
        sol, branches = sdb.search("hybrid")
        end = time.time()
        total_time = end - start
        total_solver_timed += total_time
        total_solver_branches += sdb.callCount()

    print("Finished index ", index)
    times_naive[index] = total_solver_timed / len(boardSet)
    branches_naive[index] = total_solver_branches / len(boardSet)

print("Naive done")



x_indices = np.arange(len(times_cnn))

plt.figure(figsize=(12, 8))

# plt.plot(x_indices, times_cnn, color="blue", label="cnn")
# plt.plot(x_indices, times_hybrid, color='red', label="hybrid")
# plt.plot(x_indices, times_naive, color='green', label="naive")
# plt.plot(x_indices, times_cnn, color='blue', label="cnn")

plt.plot(x_indices, branches_cnn, color = "red", label = "cnn")
plt.plot(x_indices, branches_hybrid, color = "blue", label = "hybrid")
plt.plot(x_indices, branches_naive, color = "green", label = "naive")
plt.legend()
plt.xlabel("Index")
plt.ylabel("Time")

plt.grid(True, alpha=0.5)

plt.show()






