import pandas as pd
import numpy as np
import csv
# Read only first 100,000 rows and only the first column
sudoku_file = "C:\\Users\\samgr\\PycharmProjects\\sudoku-cp-ai\\sudoku.csv"
df = pd.read_csv(sudoku_file, nrows=100_000)
boards = df.iloc[:, 0].astype(str).tolist()

print(f"Loaded {len(boards)} Sudoku boards.")
print("Example board:", boards[0])

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

def choose_next_cell(board):
    possibilities = get_possible_values(board)
    # Ignore filled cells
    possibilities[possibilities == 0] = 10
    # Find the cell with the smallest domain
    flat_index = np.argmin(possibilities)
    return flat_index

labels = []
comma_boards = []
for board in boards:
    label = choose_next_cell(board)
    labels.append(label)
    comma_board = ",".join(list(board.strip()))
    comma_boards.append(comma_board)

out_df = pd.DataFrame(
    {
        "board": comma_boards,
        "label": labels,
    }
)

with open("row_data.csv", "w") as f:
    f.write("board,label\n")  # header
    for b, l in zip(boards, labels):
        # insert commas between every character in the board
        board_str = ",".join(b)
        f.write(f"{board_str},{l}\n")
