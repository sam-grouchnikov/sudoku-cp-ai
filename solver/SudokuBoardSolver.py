import pandas as pd
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from sympy.polys import domains


from CNNGuider.model import SudokuLightning

def get_possible_vals_from_domain(domain_vec):
    return [i + 1 for i, v in enumerate(domain_vec) if v == 1]

def naive_search(board, domainStore):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return [r, c]
    return None

def hybrid_mrv(board, domainStore):
    """
    Hybrid MRV + Degree heuristic:
    - MRV: choose the empty cell with the smallest remaining domain.
    - Degree: break ties by choosing the cell that constrains the most other empty cells.
    """
    min_domain_size = 10
    candidates = []

    # Step 1: Find all empty cells and track those with the smallest domain size
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                domain_size = sum(domainStore[r][c])
                if domain_size < min_domain_size:
                    min_domain_size = domain_size
                    candidates = [(r, c)]
                elif domain_size == min_domain_size:
                    candidates.append((r, c))

    # If no empty cells, board is filled
    if not candidates:
        return None

    # Step 2: If tie, apply degree heuristic — pick cell with most constraints
    best_cell = None
    max_degree = -1
    for (r, c) in candidates:
        row_empty = sum(board[r][cc] == 0 for cc in range(9)) - 1
        col_empty = sum(board[rr][c] == 0 for rr in range(9)) - 1
        sr, sc = (r // 3) * 3, (c // 3) * 3
        grid_empty = np.sum(np.array(board)[sr:sr+3, sc:sc+3] == 0) - 1
        degree = row_empty + col_empty + grid_empty
        if degree > max_degree:
            max_degree = degree
            best_cell = (r, c)

    return best_cell

def isSolved(board):
    # Check rows
    for r in range(9):
        row_vals = [x for x in board[r] if x != 0]
        if len(row_vals) != len(set(row_vals)):
            return False

    # Check columns
    for c in range(9):
        col_vals = [board[r][c] for r in range(9) if board[r][c] != 0]
        if len(col_vals) != len(set(col_vals)):
            return False

    # Check 3x3 grids
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



class SudokuBoard:
    def __init__(self, board_string):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.board_string = board_string
        flat_array = np.array([int(char) for char in board_string])
        self.board = flat_array.reshape(9, 9).tolist()
        self.domainStore = [[[1 for _ in range(9)] for _ in range(9)] for _ in range(9)]
        self.initializeDomains()
        self.recursiveCalls = 0

        # ckpt_path = "C:\\Users\\samgr\\PycharmProjects\\sudoku-cp-ai\\solver\\row_ckpt.ckpt"
        ckpt_path = "/home/sam/sudoku/sudoku-cp-ai/row_ckpt.ckpt"
        self.model = SudokuLightning.load_from_checkpoint(ckpt_path).to(self.device)

    def predict(self, board):
        self.model.eval()
        board_array = torch.tensor(board, dtype=torch.float32, device = self.device).view(1, 1, 9, 9)

        with torch.no_grad():
            logits = self.model(board_array)
            preds = logits.view(-1).argmax()

        row, col = divmod(preds.item(), 9)
        return [row, col]

    def getValue(self, r, c):
        return self.board[r][c]

    def getBoard(self):
        return self.board

    def initializeDomains(self):
        for r in range(9):
            for c in range(9):
                val = self.board[r][c]
                if val != 0:
                    for d in range(9):
                        self.domainStore[r][c][d] = 0
                    self.domainStore[r][c][val - 1] = 1
            self.propagateRows(r)
            self.propagateCols(r)
            self.propagateGrids(r)

    def getDomainStore(self):
        return self.domainStore

    def propagate(self):
        for r in range(9):
            self.propagateRows(r)
            self.propagateCols(r)
            self.propagateGrids(r)

    def propagateRows(self, r):
        filledVals = [self.board[r][c] for c in range(9) if self.board[r][c] != 0 ]

        for c in range(9):
            cellVal = self.board[r][c]
            if cellVal != 0:
                for i in range(9):
                    self.domainStore[r][c][i] = 1 if i == cellVal - 1 else 0

            else:
                for i in range(9):
                    if (i + 1) in filledVals:
                        self.domainStore[r][c][i] = 0

    def propagateCols(self, colIdx):
        filledVals = [self.board[r][colIdx] for r in range(9) if self.board[r][colIdx] != 0]

        for rowIdx in range(9):
            cellVal = self.board[rowIdx][colIdx]
            if cellVal != 0:
                for i in range(9):
                    self.domainStore[rowIdx][colIdx][i] = 1 if i == cellVal - 1 else 0
            else:
                for i in range(9):
                    if (i + 1) in filledVals:
                        self.domainStore[rowIdx][colIdx][i] = 0

    def propagateGrids(self, gridIdx):
        startRow = (gridIdx // 3) * 3
        startCol = (gridIdx % 3) * 3

        filledVals = []
        for r in range(startRow, startRow + 3):
            for c in range(startCol, startCol + 3):
                val = self.board[r][c]
                if val != 0:
                    filledVals.append(val)

        for r in range(startRow, startRow + 3):
            for c in range(startCol, startCol + 3):
                cellVal = self.board[r][c]
                if cellVal != 0:
                    for i in range(9):
                        self.domainStore[r][c][i] = 1 if i == cellVal - 1 else 0
                else:
                    for i in range(9):
                        if (i + 1) in filledVals:
                            self.domainStore[r][c][i] = 0

    def apply_assignment_and_propagate(self, board, domainStore, r, c, value):
        board = copy.deepcopy(board)
        domainStore = copy.deepcopy(domainStore)

        board[r][c] = value

        for d in range(9):
            domainStore[r][c][d] = 1 if d == (value - 1) else 0

        queue = [(r, c)]

        while queue:
            items = queue.pop()
            cr, cc = items[0], items[1]
            assigned_vec = domainStore[cr][cc]
            try:
                assigned_val = assigned_vec.index(1) + 1
            except ValueError:
                return None, None

            for col in range(9):
                if col == cc:
                    continue
                if board[cr][col] == 0 and domainStore[cr][col][assigned_val - 1] == 1:
                    domainStore[cr][col][assigned_val - 1] = 0
                    if sum(domainStore[cr][col]) == 0:
                        return None, None
                    if sum(domainStore[cr][col]) == 1:
                        new_val = domainStore[cr][col].index(1) + 1
                        board[cr][col] = new_val
                        queue.append((cr, col))
                        pass

            for row in range(9):
                if row == cr:
                    continue
                if board[row][cc] == 0 and domainStore[row][cc][assigned_val - 1] == 1:
                    domainStore[row][cc][assigned_val - 1] = 0
                    if sum(domainStore[row][cc]) == 0:
                        return None, None
                    if sum(domainStore[row][cc]) == 1:
                        new_val = domainStore[row][cc].index(1) + 1
                        board[row][cc] = new_val
                        queue.append((row, cc))
                        pass

            startRow = (cr // 3) * 3
            startCol = (cc // 3) * 3
            for rr in range(startRow, startRow + 3):
                for cc2 in range(startCol, startCol + 3):
                    if rr == cr and cc2 == cc:
                        continue
                    if board[rr][cc2] == 0 and domainStore[rr][cc2][assigned_val - 1] == 1:
                        domainStore[rr][cc2][assigned_val - 1] = 0
                        if sum(domainStore[rr][cc2]) == 0:
                            return None, None
                        if sum(domainStore[rr][cc2]) == 1:
                            new_val = domainStore[rr][cc2].index(1) + 1
                            board[rr][cc2] = new_val
                            queue.append((rr, cc2))
                            pass

        return board, domainStore

    def search(self, method, board=None, domainStore=None, branch_sizes=None, depth=0, parent=None):

        if branch_sizes is None:
            branch_sizes = []
        if board is None:
            board = copy.deepcopy(self.board)
        if domainStore is None:
            domainStore = copy.deepcopy(self.domainStore)

        if all(all(val != 0 for val in row) for row in board):
            if isSolved(board):
                branch_sizes.append((depth, 0))
                return board, branch_sizes
            else:
                return None, branch_sizes

        import time

        if method == "cnn":
            nr, nc = self.predict(board)
        elif method == "hybrid":

            nr, nc = hybrid_mrv(board, domainStore)
        else:
            nr, nc = naive_search(board, domainStore)

        if board[nr][nc] != 0:
            nr, nc = hybrid_mrv(board, domainStore)

        cell_domain = domainStore[nr][nc]
        possible_values = get_possible_vals_from_domain(cell_domain)

        branch_sizes.append((depth, len(possible_values)))

        for val in possible_values:

            node_label = f"({nr},{nc})={val}"

            new_board, new_domain = self.apply_assignment_and_propagate(board, domainStore, nr, nc, val)
            if new_board is None:
                continue
            self.recursiveCalls += 1
            result, branch_sizes = self.search(method, new_board, new_domain, branch_sizes, depth + 1, node_label)
            if result is not None:
                return result, branch_sizes

        return None, branch_sizes

    def is_valid_assignment(self, board, r, c, val):
        # Row check
        if val in board[r]:
            return False

        # Column check
        for i in range(9):
            if board[i][c] == val:
                return False

        # 3x3 grid check
        start_row = (r // 3) * 3
        start_col = (c // 3) * 3
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == val:
                    return False

        return True

    def print(self):
        print("Board:")
        print("    " + " ".join(f"C{c}" for c in range(9)))
        for r in range(9):
            row_vals = " ".join(str(self.board[r][c]) for c in range(9))
            print(f"R{r}  {row_vals}")

    def callCount(self):
        return self.recursiveCalls