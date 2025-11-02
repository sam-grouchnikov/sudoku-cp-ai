from solver.SudokuBoardSolver import SudokuBoard
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import networkx as nx

def isSolved(board):
    for r in range(9):
        row_vals = [x for x in board[r] if x != 0]
        if len(row_vals) != len(set(row_vals)):
            return False

    for c in range(9):
        col_vals = [board[r][c] for r in range(9) if board[r][c] != 0]
        if len(col_vals) != len(set(col_vals)):
            return False

    for gr in range(3):
        for gc in range(3):
            sr, sc = gr * 3, gc * 3
            grid_vals = []
            for r in range(sr, sr + 3):
                for c in range(sc, sc + 3):
                    if board[r][c] != 0:
                        grid_vals.append(board[r][c])
            if len(grid_vals) != len(set(grid_vals)):
                return False

    return True

sb = SudokuBoard("board.csv")
sb.print()
start = time.time()
solution, branch_sizes= sb.search()
end = time.time()
total = end - start
print("Time: ", total)
if solution:
    print("\nSolved Board:")
    for row in solution:
        print(row)
else:
    print("No solution found.")

print("Solver check: ", isSolved(solution))

branch_dict = defaultdict(list)
for depth, branches in branch_sizes:
    branch_dict[depth].append(branches)


depths = sorted(branch_dict.keys())
avg_branches = [sum(branch_dict[d])/len(branch_dict[d]) for d in depths]
max_branches = [max(branch_dict[d]) for d in depths]
print(branch_dict)

plt.figure(figsize=(10,6))
plt.plot(depths, avg_branches, marker='o', label='Average Branches')
plt.plot(depths, max_branches, marker='x', linestyle='--', label='Max Branches')
plt.xlabel("Recursion Depth")
plt.ylabel("Number of Branches at Node")
plt.title("Sudoku Solver Recursion Branch Sizes (Aggregated)")
plt.legend()
plt.ylim(0, 4)
plt.grid(True)
plt.show()





