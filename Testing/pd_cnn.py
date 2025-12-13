import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

from solver.SudokuBoardSolver import SudokuBoard

sudoku_file = "sudoku_board_samples.csv"
df = pd.read_csv(sudoku_file)
boards = df.iloc[:, 0].astype(str).tolist()

board_sorted = [[] for _ in range(81)]

for board in boards:
    new_board = board.replace(".", "0")
    count = new_board.count("0")
    board_sorted[count].append(new_board)



times_1 = [0 for _ in range(81)]
branches_1 = [0 for _ in range(81)]

times_2 = [0 for _ in range(81)]
branches_2 = [0 for _ in range(81)]

times_4 = [0 for _ in range(81)]
branches_4 = [0 for _ in range(81)]

times_8 = [0 for _ in range(81)]
branches_8 = [0 for _ in range(81)]

global_board = 0

for index, boardSet in enumerate(board_sorted):
    if len(boardSet) == 0:
        continue
    total_solver_timed = 0
    total_solver_branches = 0
    for board in boardSet:
        sdb = SudokuBoard(board, "cnn_2")
        start = time.time()
        sol, branches = sdb.search("cnn")
        end = time.time()
        total_time = end - start
        total_solver_timed += total_time
        total_solver_branches += sdb.callCount()
        global_board += 1
        print("Board: ", global_board)

    print("Finished index ", index)
    times_1[index] = total_solver_timed / len(boardSet)
    branches_1[index] = total_solver_branches / len(boardSet)


x_indices = np.arange(len(times_1))

t1, t2, t4, t8 = np.array(times_1), np.array(times_2), np.array(times_4), np.array(times_4)
b1, b2, b4, b8 = np.array(branches_1), np.array(branches_2), np.array(branches_4), np.array(branches_8)
# data = {"t1": t1, "t2": t2, "t4": t4, "t8": t8, "b1": b1, "b2": b2, "b4": b4, "b8": b8}
data = {"time": t1, "branches": b1}
df = pd.DataFrame(data)
df.to_csv("data/cnn-r/performance_data_cnn-2-fr.csv")

plt.figure(figsize=(12, 8))

plt.plot(x_indices, t1, color='red', label="cnn1")
plt.plot(x_indices, b1, color='red', label="cnn1")

# plt.plot(x_indices, t2, color='green', label="cnn2")
# plt.plot(x_indices, t4, color='blue', label="cnn4")
# plt.plot(x_indices, t8, color='black', label="cnn8")

plt.legend()
plt.xlabel("Index")
plt.ylabel("Time")

plt.grid(True, alpha=0.5)

plt.show()





