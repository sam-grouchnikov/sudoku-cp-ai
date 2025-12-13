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



times_hybrid = [0 for _ in range(81)]
branches_hybrid = [0 for _ in range(81)]

global_board = 0



for index, boardSet in enumerate(board_sorted):
    if len(boardSet) == 0:
        continue
    total_solver_timed = 0
    total_solver_branches = 0
    for board in boardSet:
        sdb = SudokuBoard(board, None)
        start = time.time()
        sol, branches = sdb.search("hybrid")
        end = time.time()
        total_time = end - start
        total_solver_timed += total_time
        total_solver_branches += sdb.callCount()
        global_board += 1
        print("Board: ", global_board)

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
        sdb = SudokuBoard(board, None)
        start = time.time()
        sol, branches = sdb.search("naive")
        end = time.time()
        total_time = end - start
        total_solver_timed += total_time
        total_solver_branches += sdb.callCount()
        global_board += 1
        print("Board: ", global_board)

    print("Finished index ", index)
    times_naive[index] = total_solver_timed / len(boardSet)
    branches_naive[index] = total_solver_branches / len(boardSet)

print("Naive done")


print("Hybrid:", times_hybrid)
print("Naive:", times_naive)

print("Hybrid:", branches_hybrid)
print("Naive:", branches_naive)

x_indices = np.arange(len(times_hybrid))

th, tn = np.array(times_hybrid), np.array(times_naive)
bh, bn = np.array(branches_hybrid), np.array(branches_naive)

data = {"th": th, "tn": tn, "bh": bh, "bn": bn}
df = pd.DataFrame(data)
df.to_csv("data/performance_data_baseline.csv")

plt.figure(figsize=(12, 8))

plt.plot(x_indices, times_hybrid, color='red', label="hybrid")
plt.plot(x_indices, times_naive, color='green', label="naive")

# plt.plot(x_indices, branches_cnn, color = "red", label = "cnn")
# plt.plot(x_indices, branches_hybrid, color = "blue", label = "hybrid")
# plt.plot(x_indices, branches_naive, color = "green", label = "naive")
plt.legend()
plt.xlabel("Index")
plt.ylabel("Time")

plt.grid(True, alpha=0.5)

plt.show()





