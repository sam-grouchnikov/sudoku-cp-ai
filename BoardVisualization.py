import pandas as pd
import numpy as np
import os

def print_board(board_flat, highlight=None):
    board = np.array(board_flat).reshape(9, 9)
    print("\nSudoku Board:")
    for r in range(9):
        row_str = ""
        for c in range(9):
            val = board[r, c] if board[r, c] != 0 else "."
            if highlight == (r, c):
                row_str += f"[{val}]"
            else:
                row_str += f" {val} "
            if c % 3 == 2 and c != 8:
                row_str += "|"
        print(row_str)
        if r % 3 == 2 and r != 8:
            print("-"*33)

def human_label_sudoku_resume(raw_csv_path, labeled_csv_path, progress_path="progress.txt"):
    # Load CSV
    raw_df = pd.read_csv(raw_csv_path, nrows = 10000)  # use header=None for raw 81-char strings
    if raw_df.shape[1] > 1:
        raw_df = raw_df.drop(raw_df.columns[1], axis=1)

    labeled_rows = []

    # Determine starting index
    start_idx = 0
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r") as f:
                start_idx = int(f.read().strip())
            print(f"Resuming from row {start_idx}")
        except:
            start_idx = 0

    for idx in range(start_idx, len(raw_df)):
        board_str = str(raw_df.iloc[idx, 0])
        if len(board_str) != 81:
            print(f"Skipping row {idx}: not 81 characters")
            continue

        # Convert to integers safely
        try:
            board_flat = [int(c) for c in board_str]
        except ValueError:
            print(f"Skipping row {idx}: contains non-integer characters")
            continue

        print_board(board_flat)

        # Ask for human input
        while True:
            user_input = input("Next cell to choose? Enter row,col (0-indexed): ")
            try:
                r, c = map(int, user_input.strip().split(","))
                if 0 <= r < 9 and 0 <= c < 9 and board_flat[r*9 + c] == 0:
                    break
                else:
                    print("Invalid input: cell must be empty and within 0-8.")
            except:
                print("Invalid input. Format: row,col")

        next_cell_index = r*9 + c
        labeled_rows.append(board_flat + [next_cell_index])

        # Save labeled CSV after each entry
        labeled_df = pd.DataFrame(labeled_rows)
        labeled_df.to_csv(labeled_csv_path, index=False, header=False)

        # Save progress
        with open(progress_path, "w") as f:
            f.write(str(idx + 1))

    print(f"Labeled dataset saved to {labeled_csv_path}")
    print("All boards processed!")

# Example usage:
human_label_sudoku_resume("sudoku.csv", "labeled_sudoku.csv")
