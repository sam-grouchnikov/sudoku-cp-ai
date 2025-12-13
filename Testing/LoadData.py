import pandas as pd

df = pd.read_csv("sudoku_boards_sorted.csv")

# Group by zero_count
groups = df.groupby("zero_count")


rows = []

for zero_count, group in groups:
    if len(group) <= 100:
        # Take all examples
        rows.append(group)
    else:
        rows.append(group.iloc[:100])

# Combine all collected rows
sampled = pd.concat(rows, ignore_index=True)

# Save to a new file
sampled.to_csv("sudoku_board_samples.csv", index=False, header=False)