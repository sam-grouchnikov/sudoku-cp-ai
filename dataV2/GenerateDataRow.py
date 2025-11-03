import pandas as pd
import numpy as np
import csv
# Read only first 100,000 rows and only the first column
sudoku_file = "C:\\Users\\samgr\\PycharmProjects\\sudoku-cp-ai\\sudoku.csv"
df = pd.read_csv(sudoku_file, nrows=1_000_000)
boards = df.iloc[:, 0].astype(str).tolist()

print(f"Loaded {len(boards)} Sudoku boards.")
# print("Example board:", boards[0])

def get_possible_values(board):
    board = np.array(list(map(int, board))).reshape(9, 9)
    possibilities = np.zeros((9, 9), dtype=int)

    for r in range(9):
        for c in range(9):
            if board[r, c] != 0:
                possibilities[r, c] = 0  # already filled
                continue

            row_vals = set(board[r, :])
            col_vals = set(board[:, c])
            grid_vals = set(board[r//3*3:r//3*3+3, c//3*3:c//3*3+3].flatten())
            used = row_vals | col_vals | grid_vals
            possibilities[r, c] = 9 - len(used - {0})  # domain size

    return possibilities

# print(get_possible_values(boards[0]))

def choose_next_cell_mrv(board):
    possibilities = get_possible_values(board)
    possibilities[possibilities == 0] = 10
    flat_index = np.argmin(possibilities)
    return flat_index

def choose_next_cell_mrv_hybrid(board):


    possibilities = get_possible_values(board)

    board = np.array(list(map(int, board))).reshape(9, 9)

    # Ignore filled cells
    possibilities[possibilities == 0] = 10
    # Find the cell with the smallest domain
    min_val = possibilities.min()
    candidates = np.argwhere(possibilities == min_val)

    if len(candidates) == 1:
        return candidates[0][0] * 9 + candidates[0][1]

    max_deg = -1
    best_cell = None

    for r, c in candidates:
        row_empty = np.sum(board[r, :] == 0) - 1
        col_empty = np.sum(board[:, c] == 0) - 1
        sr, sc = (r // 3) * 3, (c // 3) * 3
        grid_empty = np.sum(board[sr:sr+3, sc:sc+3] == 0) - 1
        degree = row_empty + col_empty + grid_empty
        if degree > max_deg:
            max_deg = degree
            best_cell = (r, c)

    return best_cell[0] * 9 + best_cell[1]

# print(choose_next_cell_mrv(boards[2]))
# print(choose_next_cell_mrv_hybrid(boards[2]))


labels = []
boards_str = []

for board in boards:
    label = choose_next_cell_mrv_hybrid(board)
    labels.append(label)
    # remove whitespace and keep the board as a single 81-character string
    boards_str.append(board.strip())

# make DataFrame with only 2 columns
out_df = pd.DataFrame({
    "board": boards_str,
    "label": labels,
})

with open("row_data.csv", "w") as f:
    f.write("board,label\n")
    for b, l in zip(boards, labels):
        f.write(f"{b},{l}\n")
